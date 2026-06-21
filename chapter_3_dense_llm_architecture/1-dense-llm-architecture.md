# Chapter 3: Dense LLM Architecture

To scale up from our character-level toy transformer to a production-grade **Large Language Model (LLM)**, we do not simply stack standard transformer layers. Standard layers (like the 2017 Vaswani design) introduce unnecessary computation and memory overheads that bottleneck large-scale training and inference. 

Modern **Dense LLMs** (e.g., Llama 3, Gemma, Mistral) employ a refined, hardware-optimized block anatomy. This chapter details these modern components and demonstrates how to implement them from scratch.

---

## 1. The Modern Block Anatomy

```
Standard Transformer Block (Vaswani)       Modern Dense LLM Block (Llama-style)
+------------------------------------+      +------------------------------------+
| Input                              |      | Input                              |
|   |                                |      |   |                                |
|   v                                |      |   v                                |
| [ LayerNorm ]                      |      | [ RMSNorm ]                        |
|   |                                |      |   |                                |
|   v                                |      |   v                                |
| [ Multi-Head Attention (MHA) ]     |      | [ Grouped-Query Attention (GQA) ]  |
|   | (Absolute/Sinusoidal Position) |      |   | (Rotary Position Embeddings)   |
|   v                                |      |   v                                |
| [ Residual Add ]                   |      | [ Residual Add ]                   |
|   |                                |      |   |                                |
|   v                                |      |   v                                |
| [ LayerNorm ]                      |      | [ RMSNorm ]                        |
|   |                                |      |   |                                |
|   v                                |      |   v                                |
| [ Feed-Forward MLP (GELU/ReLU) ]   |      | [ SwiGLU Feed-Forward MLP ]        |
|   |                                |      |   |                                |
|   v                                |      |   v                                |
| [ Residual Add ] ---> Output       |      | [ Residual Add ] ---> Output       |
+------------------------------------+      +------------------------------------+
```

---

## 2. Component Deep Dive

### A. Root Mean Square Normalization (RMSNorm)
Standard `LayerNorm` normalizes inputs by calculating both their mean ($\mu$) and variance ($\sigma^2$):
$$\text{LN}(x) = \frac{x - \mu}{\sigma} \odot \gamma + \beta$$

RMSNorm simplifies this process by assuming that centering the inputs (subtracting the mean $\mu$) is computationally redundant for deep network stabilization. Instead, it scales by the root mean square (RMS):
$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot \gamma \quad \text{where} \quad \text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}$$

* **Why it matters for Inference:** Computing $\mu$ requires a sum and a division across the vector dimension, followed by subtracting it from each element. Skipping $\mu$ reduces the number of high-latency GPU global memory access cycles (reads and writes) by $\approx 30\%$, enabling fast fused CUDA normalization kernels.

---

### B. Rotary Position Embedding (RoPE)
In self-attention, we project tokens into query ($Q$) and key ($K$) states. Absolute position embeddings add positional vectors directly to input embeddings. 

RoPE instead applies a rotation matrix to the projected query and key states at each layer. For any two-dimensional chunk of a vector at position $m$, we rotate it in the complex plane by an angle proportional to the position $m$:
$$R_{\Theta, m}^2 \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$$
where $\theta_i = 10000^{-2(i-1)/d}$.

By rotating $Q$ and $K$ vectors, the dot-product attention score naturally encapsulates the **relative distance** between tokens:
$$\langle R_{\Theta, m}^d q, R_{\Theta, n}^d k \rangle = q^T R_{\Theta, n-m}^d k$$

```
   Token Position m (Query)                  Token Position n (Key)
            \                                       /
             \  (Rotate by m*theta)                /  (Rotate by n*theta)
              v                                   v
         [ Rotated Q ] --------------------> [ Rotated K ]
                       Dot-product score is proportional
                       to relative offset angle (n - m)
```

* **Why it matters for Inference:** Models trained with RoPE exhibit superior length extrapolation. The model's context window can be extended at inference time (using techniques like RoPE scaling/interpolation) without needing to re-train absolute positional lookup matrices.

---

### C. Grouped-Query Attention (GQA)
Autoregressive decoding is heavily bottlenecked by VRAM capacity and memory bandwidth because the GPU must read and cache Key-Value (KV) projections for all past tokens (the KV Cache) at every step.

* **Multi-Head Attention (MHA):** Every Query head has its own Key and Value head. (High VRAM footprint).
* **Multi-Query Attention (MQA):** All Query heads share a single Key and Value head. (Low VRAM footprint, but hurts model capacity).
* **Grouped-Query Attention (GQA):** A middle ground. Query heads are partitioned into groups, and each group shares a single Key-Value head.

```
 MHA (Multi-Head)              GQA (Grouped-Query)            MQA (Multi-Query)
Q Q Q Q Q Q Q Q                 Q Q Q Q Q Q Q Q                Q Q Q Q Q Q Q Q
| | | | | | | |                 \ \ / / \ \ / /                \ \ \ / / / / /
v v v v v v v v                  v   v   v   v                      v
K K K K K K K K                  K   K   K   K                      K
V V V V V V V V                  V   V   V   V                      V
```

If we have $H_q$ Query heads and $H_{kv}$ Key-Value heads, the GQA group size is $G = H_q / H_{kv}$. During computation, keys and values are repeated (broadcast) $G$ times to match query shapes.

* **Why it matters for Inference:** If $H_{kv} = H_q / 8$, GQA cuts KV Cache memory consumption by **$8\times$**. This directly scales the maximum serving concurrency (batch size) and prevents high-concurrency requests from hitting VRAM limits.

---

### D. SwiGLU Activation
Standard Feed-Forward networks project input $x$ through a linear layer, apply an activation (like GELU), and project it back:
$$\text{FFN}(x) = \text{GELU}(x W_1) W_2$$

SwiGLU replaces this with a Gated Linear Unit (GLU) using the Swish (SiLU) activation function:
$$\text{SwiGLU}(x) = \left( \text{SiLU}(x W_{\text{gate}}) \cdot x W_{\text{up}} \right) W_{\text{down}}$$

```
               Input (x)
              /         \
             /           \
     [ Linear Gate ]   [ Linear Up ]
            |                 |
      [ SiLU/Swish ]          |
             \               /
              \             /
             [ Elementwise * ]
                    |
             [ Linear Down ]
                    |
                 Output
```

* **Why it matters:** SwiGLU MLPs feature an additional projection matrix ($W_{\text{gate}}$), increasing parameter count and FLOPs for the same hidden size. However, this gating mechanism allows the network to learn complex non-linear combinations more expressively, improving parameter efficiency and stabilizing convergence.
