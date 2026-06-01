# What are Transformers?

The transformer architecture, introduced by Vaswani et al. in the landmark 2017 paper [Attention is All You Need](https://arxiv.org/pdf/1706.03762), is the foundation of modern Large Language Models (LLMs). By replacing recurrent and convolutional structures with self-attention, transformers unlocked massive parallelization and enabled the training of models with hundreds of billions of parameters.

---

## 1. The Paradigm Shift: From RNNs to Self-Attention

Before the transformer, sequence modeling relied primarily on **Recurrent Neural Networks (RNNs)** (including LSTMs and GRUs) and **Convolutional Neural Networks (CNNs)**. Both architectures have inherent limitations that bottlenecked both training and inference.

### Recurrent Neural Networks (RNNs)
* **Sequential Bottleneck**: RNNs process tokens sequentially, one after another. To compute the hidden state at step $t$, the network must wait for the hidden state at step $t-1$. This sequential dependency prevents training from being parallelized across the sequence dimension.
* **Vanishing/Exploding Gradients**: As sequences grow longer, gradients backpropagated through time tend to vanish or explode, making it difficult for RNNs to learn long-term dependencies.
* **Information Bottleneck**: Classic encoder-decoder RNNs compress the entire input sequence into a single fixed-size context vector, leading to information loss for long sequences.

### Convolutional Neural Networks (CNNs)
* **Limited Receptive Field**: While CNNs can process tokens in parallel, their receptive field is limited by the kernel size. Capturing long-range dependencies requires stacking many layers, which increases computational depth and memory footprints.

### The Transformer Solution
The transformer architecture discards recurrence and convolution entirely. It uses **self-attention** to allow every token in a sequence to connect directly to every other token. This architecture provides two main advantages:
1. **$O(1)$ Path Length**: The maximum path length between any two tokens is $O(1)$, which eliminates the vanishing gradient problem over long sequences.
2. **Massive Parallelization**: Because there are no sequential dependencies between steps during training, the entire sequence can be processed simultaneously, maximizing GPU tensor core utilization.

---

## 2. Historical Timeline & Evolution

The development of the transformer was a gradual evolution driven by the search for better alignment in machine translation.

* **Bahdanau Attention (2014)**: Pioneered by Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. In sequence-to-sequence RNNs, they introduced a mechanism that allowed the decoder to search through the encoder's hidden states at each step. Instead of relying on a single bottleneck context vector, the decoder computed an "attention score" over all input states, focusing on the most relevant parts of the input sequence.
* **Google's GNMT (2016)**: Google deployed the Attention-based LSTM in Google Neural Machine Translation (GNMT), achieving major quality improvements. However, the architecture still relied on recurrence, meaning training times remained long and hardware utilization was low.
* **The Transformer (2017)**: The paper "Attention is All You Need" demonstrated that attention alone, without any recurrent layers, was sufficient to achieve state-of-the-art results. This simplified the architecture and allowed hardware to run calculations in parallel.

---

## 3. The Core Engine: Self-Attention Mechanics

Self-attention allows the model to dynamically compute the relationship between each token in a sequence. 

### Query, Key, and Value Vectors
For an input sequence of vectors, the model projects each token vector into three separate representations using learned weight matrices:
* **Query ($Q$)**: Represents the token looking for context.
* **Key ($K$)**: Represents the token offering context to other tokens.
* **Value ($V$)**: Represents the actual information content of the token.

### Scaled Dot-Product Attention
The relationship (similarity score) between queries and keys is calculated using a dot product. The full mathematical formulation is:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
* $QK^T$ represents the raw attention scores between all pairs of tokens.
* $d_k$ is the dimension of the key vectors.
* $\sqrt{d_k}$ is the scaling factor. This scaling factor is critical: without it, for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions with extremely small gradients (vanishing gradients during training).
* $\text{softmax}$ normalizes the scores into a probability distribution.
* Multiplying by $V$ computes a weighted sum of the values, where the weights represent how much attention the query token should pay to every other token.

> [!NOTE]
> **Inference Engineering Insight**: The calculation of $QK^T$ has a quadratic computational complexity ($O(N^2)$) relative to the sequence length $N$. This quadratic bottleneck is one of the main challenges in deploying long-context LLMs.

### Multi-Head Attention (MHA)
Instead of performing a single attention function, Multi-Head Attention projects $Q$, $K$, and $V$ vectors into multiple low-dimensional subspaces. The model runs the attention mechanism in parallel across these "heads," allowing it to attend to information from different representation subspaces at different positions simultaneously.

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$
$$\text{where} \quad \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

---

## 4. Comparing Attention Variants: Classical Attention, Self-Attention, and Cross-Attention

While all attention mechanisms share the core objective of dynamically routing information across a sequence, they differ significantly in **where** their Query ($Q$), Key ($K$), and Value ($V$) vectors originate.

| Attention Variant | Query Source ($Q$) | Key Source ($K$) | Value Source ($V$) | Typical Architectural Context |
| :--- | :--- | :--- | :--- | :--- |
| **Classical Attention** | Current Decoder RNN hidden state ($h_t^{\text{dec}}$) | All Encoder RNN hidden states ($H^{\text{enc}}$) | All Encoder RNN hidden states ($H^{\text{enc}}$) | Seq2Seq Recurrent Networks (e.g., LSTMs) |
| **Self-Attention** | Same input sequence ($X$) | Same input sequence ($X$) | Same input sequence ($X$) | Transformers (Encoder or Decoder blocks) |
| **Cross-Attention** | Different sequence (Decoder hidden states $H^{\text{dec}}$) | Source sequence (Encoder representations $H^{\text{enc}}$) | Source sequence (Encoder representations $H^{\text{enc}}$) | Transformer Encoder-Decoder bottlenecks |

---

### A. Classical Attention (Seq2Seq / Recurrent Attention)
Introduced as an add-on to traditional RNN structures, classical attention helps bridge the representation bottleneck between an encoder and a decoder.
* **Mechanism**: The query $Q$ represents the current translation token's context (from the decoder RNN state at time $t$). The keys $K$ and values $V$ represent the source sentence contexts (the list of all hidden states generated by the encoder RNN).
* **Source**: $Q$ comes from sequence B (decoder), while $K$ and $V$ come from sequence A (encoder).
* **Primary Use Cases**:
  * **Recurrent Machine Translation**: Original LSTM-based translation networks (such as Google Neural Machine Translation).
  * **Classic Speech-to-Text**: Recurrent neural networks mapping acoustic frames to character/word sequences.

### B. Self-Attention (Intra-Attention)
Self-attention relates different positions of a single sequence to compute a representation of the very same sequence.
* **Mechanism**: $Q$, $K$, and $V$ are all projected from the same input sequence embedding matrix. For example, when processing a sentence, each word queries all other words in the same sentence to learn contextual relations (e.g., resolving pronouns).
* **Source**: $Q$, $K$, and $V$ all originate from the same input sequence.
* **Primary Use Cases**:
  * **Autoregressive Language Modeling**: Large Language Models (such as Llama, GPT, or Mistral) predicting the next token by looking back at the prompt and previously generated tokens.
  * **Contextual Embedding Generation**: Models like BERT learning bi-directional representations of text by letting every token attend to all other tokens.
  * **Vision Transformers (ViT)**: Image patches attending to other patches in the same image to recognize objects.

### C. Cross-Attention (Encoder-Decoder Attention)
Cross-attention connects two different sequences, allowing information to flow from one domain or representation space to another.
* **Mechanism**: The queries $Q$ come from the target sequence (the decoder block's current representations). The keys $K$ and values $V$ are imported from a different source sequence (usually the final outputs of an encoder block).
* **Source**: $Q$ comes from one sequence, while $K$ and $V$ come from a completely separate sequence.
* **Primary Use Cases**:
  * **Transformer-Based Machine Translation**: In encoder-decoder models like T5 or BART, the decoder cross-attends to the encoder's output representations to generate target-language words.
  * **Text-to-Image Generation**: In Diffusion models (such as Stable Diffusion), the image-generation U-Net cross-attends to the text embeddings from a text encoder (like CLIP) to ensure the generated image matches the prompt.
  * **Multimodal Question Answering**: Models where text queries cross-attend to visual or audio feature maps to extract relevant answers.

---

## 5. Architectural Variants

Depending on how self-attention is applied and masked, transformers are categorized into three primary structural archetypes.

### A. Encoder-Decoder (Seq2Seq Transformers)
* **Design**: Consists of two distinct blocks. The **encoder** processes the input sequence bidirectionally (tokens attend to all other tokens). The **decoder** generates the target sequence autoregressively. The decoder uses **masked self-attention** to prevent attending to future tokens, and **cross-attention** to attend to the final representations produced by the encoder.
* **Examples**: T5, BART.
* **Inference Profile**: Highly efficient for structured translation or text-to-text generation tasks, but requires managing active state across both blocks.

### B. Decoder-Only (Causal Transformers)
* **Design**: Consists of a single stack of transformer blocks. It uses **causal masking** in the self-attention layer to ensure that a token at position $t$ can only attend to tokens at positions $\le t$. This prevents the model from looking ahead during training.
* **Examples**: GPT-4, Llama, Mistral, Gemma.
* **Inference Profile**: This architecture dominates modern generative AI. During inference, it operates in two phases:
  1. **Prefill Phase**: Processes the entire prompt in parallel, computing and caching Key and Value states.
  2. **Decoding Phase**: Generates tokens autoregressively, one by one. Because keys and values from previous tokens do not change, they are stored in a **KV Cache** to avoid redundant calculations. This changes the computational complexity of generating a token from $O(N^2)$ to $O(N)$.

### C. Encoder-Only (Non-Causal Transformers)
* **Design**: Consists of a single stack of transformer blocks without masking. Every token can attend to all other tokens, regardless of their position (fully bidirectional).
* **Examples**: BERT, RoBERTa.
* **Inference Profile**: Primarily used for classification, feature extraction, and embedding generation. Since the entire sequence is processed in a single forward pass, there is no autoregressive generation phase or KV caching requirement.

---

## 6. Anatomy of a Transformer Block: Classic vs. Modern Configurations

A transformer model is not just self-attention; it consists of stacked identical blocks containing self-attention, layer normalization, residual connections, and a position-wise feed-forward network (FFN). How these components are arranged and configured inside the block defines the differences between the classic 2017 Transformer and modern LLMs.

### A. Pre-LN vs. Post-LN Layouts
The location of the Layer Normalization (LN) module dramatically affects training stability and gradient propagation.

```
Post-LN (Original Transformer):
Input ---> [ Self-Attention ] ---> (+) ---> [ LayerNorm ] ---> [ FFN ] ---> (+) ---> [ LayerNorm ] ---> Output
   |                            ^                                 |       ^
   +----------------------------+                                 +-------+

Pre-LN (Modern LLMs):
Input ---> [ LayerNorm ] ---> [ Self-Attention ] ---> (+) ---> [ LayerNorm ] ---> [ FFN ] ---> (+) ---> Output
   |                                                   ^          |                             ^
   +---------------------------------------------------+          +-----------------------------+
```

* **Post-LN (Original)**: Normalization is applied *after* residual addition. 
  $$\tilde{x} = \text{LayerNorm}(x + \text{SubLayer}(x))$$
  * *Characteristics*: Gradients in deeper layers tend to be much larger than in earlier layers, making deep networks highly unstable to train without a precise learning rate warmup schedule.
* **Pre-LN (Modern)**: Normalization is applied to the input *before* passing it to the sub-layer (Attention or FFN). 
  $$x = x + \text{SubLayer}(\text{LayerNorm}(x))$$
  * *Characteristics*: The residual stream is kept clear, allowing gradients to flow directly from the final layer back to the first layer. This makes training highly stable, allowing for deeper models without warmup-related failures.

---

### B. Normalization Modules: LayerNorm vs. RMSNorm
Modern inference optimizations prioritize reducing memory bandwidth bottlenecks.

* **LayerNorm (LN)**: Normalizes using both mean and variance:
  $$\text{LN}(x) = \frac{x - \mu}{\sigma} \odot \gamma + \beta$$
* **Root Mean Square Normalization (RMSNorm)**: Removes the mean calculation entirely and normalizes only by the root-mean-square:
  $$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot \gamma \quad \text{where} \quad \text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}$$
  * *Inference Engineering Insight*: By omitting the mean calculation ($\mu$), RMSNorm requires fewer GPU memory access cycles and reduces arithmetic operations. This makes it a popular choice in modern architectures (e.g., Llama, Mistral) to achieve higher throughput and easily fuse normalization kernels.

---

### C. FFN Activations: GELU and SwiGLU
The feed-forward network (FFN) typically accounts for roughly two-thirds of a model's total parameter count. 

* **Classic (ReLU & GELU)**: Classic transformers used `ReLU` or `GELU` (Gaussian Error Linear Unit) inside a two-layer MLP:
  $$\text{FFN}(x) = \text{Activation}(x W_1) W_2$$
* **Modern (SwiGLU)**: Standardized by PaLM, Llama, and Mistral, the SwiGLU (Swish Gated Linear Unit) replaces the basic MLP. It uses three projection matrices ($W_{\text{gate}}$, $W_{\text{up}}$, $W_{\text{down}}$) and a SiLU/Swish activation:
  $$\text{SwiGLU}(x) = \left(\text{Swish}(x W_{\text{gate}}) \otimes x W_{\text{up}}\right) W_{\text{down}}$$
  * *Inference Engineering Insight*: SwiGLU requires more parameters and FLOPS for the same hidden state size than GELU, but models converge faster and exhibit higher parameter efficiency. Consequently, hardware configurations must account for three distinct linear matrices inside the FFN rather than two.

---

### D. Positional Encoding: Absolute vs. Rotary
Because transformers process all tokens in parallel, the self-attention mechanism is permutation-invariant. Positional information must be injected explicitly.

* **Absolute Positional Encodings**: Positional vectors (sinusoidal curves or learned parameters) are added directly to the token embeddings at the very beginning of the model. 
  * *Limitation*: The model cannot easily extrapolate to sequences longer than those seen during training because the absolute vectors lose semantic meaning at unseen positions.
* **Rotary Position Embedding (RoPE)**: Applies a rotation matrix to the $Q$ and $K$ projections at each layer. It encodes relative positions by rotating the query and key vectors in two-dimensional slices of the hidden dimension by an angle proportional to their absolute positions:
  $$\langle \mathbf{R}_{\Theta, m}^d \mathbf{q}, \mathbf{R}_{\Theta, n}^d \mathbf{k} \rangle = \mathbf{q}^T \mathbf{R}_{\Theta, n-m}^d \mathbf{k}$$
  * *Inference Engineering Insight*: RoPE enables high generalization to varying context lengths. Since it is applied directly in the self-attention layer, it requires special fused GPU kernels (e.g., in FlashAttention) to rotate $Q$ and $K$ on-the-fly without creating massive intermediate tensors in memory.

---

## 7. Model Scale and Evolution: From Vaswani to Modern Giants

At a macro level, a complete transformer model is constructed by sequentially stacking multiple identical transformer blocks on top of each other. Crucially, these blocks are bounded at both the input and output boundaries by **dense (fully connected) layers**:

1. **Input Boundary (Embedding)**: Word or token indices are mapped to dense embedding vectors via a dense embedding lookup table. Positional encodings are added or applied here.
2. **Intermediate Stack**: The dense vectors flow through a stack of $N$ sequential transformer block modules, where attention and FFN sub-layers refine the representation.
3. **Output Boundary (Projection)**: The output of the final block is projected back to the target vocabulary space using a final dense linear layer (often referred to as the "unembedding layer" or `lm_head`). Applying a softmax over this output produces the probability distribution for next-token predictions.

### The Original Vaswani Configuration (2017)
The original "Attention is All You Need" paper proposed two configurations designed for machine translation:

* **Transformer-Base (65 Million Parameters)**:
  * **Architecture**: Encoder-Decoder
  * **Module/Layer Count ($N$)**: 6 encoder layers and 6 decoder layers (totaling 12 block modules).
  * **Model Dimension ($d_{\text{model}}$)**: 512
  * **FFN Hidden Dimension ($d_{\text{ff}}$)**: 2048
  * **Attention Heads ($h$)**: 8
* **Transformer-Big (213 Million Parameters)**:
  * **Architecture**: Encoder-Decoder
  * **Module/Layer Count ($N$)**: 6 encoder layers and 6 decoder layers (totaling 12 block modules).
  * **Model Dimension ($d_{\text{model}}$)**: 1024
  * **FFN Hidden Dimension ($d_{\text{ff}}$)**: 4096
  * **Attention Heads ($h$)**: 16

---

### Scaling up to Modern Models
Modern generative architectures have scaled up parameter counts and layer counts by several orders of magnitude to achieve advanced reasoning capabilities, while adapting the underlying block anatomy for inference efficiency.

| Model | Architecture | Parameter Count | Layer Count ($N$) | Model Dimension ($d_{\text{model}}$) | Key Architectural Details |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Vaswani Base (2017)** | Encoder-Decoder | 65M | 6 + 6 | 512 | Post-LN, ReLU, Absolute Positional |
| **Vaswani Big (2017)** | Encoder-Decoder | 213M | 6 + 6 | 1024 | Post-LN, ReLU, Absolute Positional |
| **BERT-Base (2018)** | Encoder-Only | 110M | 12 | 768 | Post-LN, GELU, Absolute Learned |
| **GPT-2 XL (2019)** | Decoder-Only | 1.5B | 48 | 1600 | Pre-LN, GELU, Absolute Learned |
| **GPT-3 (2020)** | Decoder-Only | 175B | 96 | 12288 | Pre-LN, GELU, Sparse Attention |
| **Llama 3 8B (2024)** | Decoder-Only | 8B | 32 | 4096 | Pre-LN, RMSNorm, SwiGLU, RoPE, GQA |
| **Llama 3 70B (2024)** | Decoder-Only | 70B | 80 | 8192 | Pre-LN, RMSNorm, SwiGLU, RoPE, GQA |
| **Llama 3 405B (2024)**| Decoder-Only | 405B | 126 | 16384 | Pre-LN, RMSNorm, SwiGLU, RoPE, GQA |


