# High-Throughput & Low-Latency LLM Agent Deployment: Tuning Qwen on vLLM & Blackwell

An engineering guide to optimizing open-weight LLM inference for asynchronous agent workloads using vLLM on modern GPU architectures.

---

## 1. Production Context & Workload Profile

Deploying Large Language Models (LLMs) for enterprise agentic workflows introduces a fundamental engineering challenge: **asynchronous, bursty query patterns with complex tool definitions**. Unlike traditional batch processing or high-concurrency chatbot applications, autonomous LLM agents spend long periods waiting for host signals, followed by abrupt executions requiring multi-step tool calls.

### The ETF Lab Deployment Topology

In our **ETF (Enterprise Testing Facility) Lab**, an infrastructure agent monitors several dozen physical and virtual hosts. The agent analyzes host telemetry (CPU spikes, memory saturation, thermal throttling, network dropouts) and issues targeted remediation commands using structured tool calling.

```
                    ┌─────────────────────────────────────────┐
                    │            Host Telemetry               │
                    │   (CPU, Memory, Network Metrics)        │
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           NVIDIA RTX 6000 Blackwell                             │
│                              (~97 GB Total VRAM)                                │
│                                                                                 │
│   ┌────────────────────────────────────────┐   ┌────────────────────────────┐   │
│   │           vLLM Engine Replica          │   │      TEI via NVIDIA MPS    │   │
│   │                                        │   │   (Text Embeddings Inference)│   │
│   │  Model: Qwen3.6-35B-A3B (FP8)          │   │                            │   │
│   │  vLLM Config:                          │   │  Allocation: ~8 GB VRAM    │   │
│   │   - max-model-len: 131,072             │   │                            │   │
│   │   - gpu-memory-utilization: 0.60       │   └────────────────────────────┘   │
│   │   - enable-prefix-caching: True        │                                    │
│   │   - enable-auto-tool-choice: True      │   ┌────────────────────────────┐   │
│   │   - tool-call-parser: True             │   │   CUDA Driver & Overhead   │   │
│   └────────────────────────────────────────┘   │        ~3-5 GB VRAM        │   │
│                                                └────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Server Configuration Summary

* **Hardware Base:** Single replica on NVIDIA RTX 6000 (Blackwell generation) featuring **~97 GB VRAM**.
* **Current Model:** `Qwen3.6-35b-a3b-fp8` (Mixture-of-Experts architecture with FP8 quantization).
* **Co-located Services:** Text Embeddings Inference (TEI) running on the same GPU via NVIDIA Multi-Process Service (MPS), consuming **~8 GB VRAM**.
* **vLLM Parameters:**
  * `max-model-len`: `131072` (128k context window capacity)
  * `gpu-memory-utilization`: `0.60` (reserving 60% of GPU memory for vLLM)
  * `max-num-seqs`: `512`
  * `enable-prefix-caching`: `True`
  * `enable-auto-tool-choice`: `True`

### The Underutilization Paradox

With `gpu-memory-utilization` set conservatively to `0.60`, vLLM allocates **~58.2 GB** out of the 97 GB total VRAM. After holding ~35 GB for FP8 model weights, the remaining KV cache memory pool is constrained. Because the monitoring agent triggers infrequently, the GPU compute engines sit idle most of the time.

This introduces a core architectural trade-off:
1. **Option A:** Increase `gpu-memory-utilization` to `0.85+` to maximize the KV cache memory pool, absorbing long context histories for host remediation sessions.
2. **Option B:** Maintain or reduce vLLM's memory footprint to co-locate a secondary model—such as a small **Draft Model** for Speculative Decoding or an auxiliary classification model.

---

## 2. Model Architecture & Numeric Precision Mechanics

To make informed tuning decisions, we must analyze the structural differences between **Qwen 3.6-35B-A3B** and the newer **Qwen 3.8-27B**, alongside the physics of low-precision floating-point formats.

### Qwen Architectural Comparison

| Metric / Parameter | Qwen 3.6-35B-A3B | Qwen 3.8-27B (Dense) |
| :--- | :--- | :--- |
| **Architecture Type** | Mixture-of-Experts (MoE) | Dense Transformer |
| **Total Parameters** | ~35 Billion | ~27 Billion |
| **Active Parameters / Token** | ~3 Billion | 27 Billion |
| **FP8 Weight Footprint** | ~35 GB VRAM | ~27 GB VRAM |
| **FP4 Weight Footprint** | N/A | ~13.5 GB VRAM |
| **Compute Characteristic** | Ultra-low compute per token, high memory bandwidth requirement to fetch routing weights | Higher compute per token, uniform weight activation across all layers |

> [!IMPORTANT]
> **MoE Memory Footprint Rule:** In Mixture-of-Experts architectures like Qwen 3.6-35B-A3B, while only ~3B parameters are activated per token (giving the latency profile of a small model), **all 35B parameters must reside in VRAM simultaneously**.

### Deep Dive: FP8 and FP4 Numeric Formats

Quantization reduces memory footprint and accelerates Tensor Core execution by reducing bit width. The representation of a floating-point number is governed by three fields:

$$\text{Value} = (-1)^{\text{sign}} \times 2^{\text{exponent} - \text{bias}} \times \left(1 + \frac{\text{mantissa}}{2^m}\right)$$

```
  FP16 (16-bit):  [ S | E E E E E | M M M M M M M M M M ]  (1 Sign, 5 Exponent, 10 Mantissa)
  FP8 E4M3:       [ S | E E E E | M M M ]                  (1 Sign, 4 Exponent, 3 Mantissa)
  FP8 E5M2:       [ S | E E E E E | M M ]                  (1 Sign, 5 Exponent, 2 Mantissa)
  FP4 E2M1:       [ S | E E | M ]                          (1 Sign, 2 Exponent, 1 Mantissa)
```

#### FP8 Variants: E4M3 vs. E5M2

1. **FP8 E4M3 (1 Sign, 4 Exponent, 3 Mantissa):**
   * **Dynamic Range:** Max value $\approx \pm 448$.
   * **Precision:** Higher numerical precision due to 3 mantissa bits.
   * **Primary Application:** Model weights ($W$) and activation tensors where high precision is required to preserve model quality.
2. **FP8 E5M2 (1 Sign, 5 Exponent, 2 Mantissa):**
   * **Dynamic Range:** Matches IEEE FP16 ($\approx \pm 57344$).
   * **Precision:** Lower precision (2 mantissa bits).
   * **Primary Application:** Key-Value (KV) Cache tensors and gradient accumulation where wide dynamic range prevents overflow during long-context attention calculations.

#### FP4 (NVFP4) Quantization Mechanics

FP4 (E2M1) reduces each parameter to 4 bits (0.5 bytes). With only 1 mantissa bit and 2 exponent bits, FP4 relies on **block-wise scaling factors**:

$$\mathbf{W}_{\text{FP16}} \approx \mathbf{S}_{\text{block}} \odot \mathbf{W}_{\text{FP4}}$$

* **VRAM Savings:** Qwen 3.8-27B in FP4 occupies only **~13.5 GB VRAM**, freeing up over **75 GB VRAM** on an RTX 6000 for KV caching or co-located models.
* **Throughput:** Blackwell GPUs feature native FP4 Tensor Cores capable of doubling math throughput relative to FP8.

---

## 3. Transformer Attention & vLLM Memory Innovations

Understanding attention mechanics is critical to solving the memory bottlenecks of long-context LLM deployment.

### The Attention Mechanism as Database Lookup

Conceptually, the attention mechanism operates as a continuous, differentiable database query:

```
          Input Embedding (X)
                 │
      ┌──────────┼──────────┐
      ▼          ▼          ▼
   Query (Q)  Key (K)    Value (V)
      │          │          │
      └────┬─────┘          │
           ▼                │
     Dot Product            │
    (Similarity)            │
           │                │
           ▼                │
   Softmax Scaling          │
     (Weights A)            │
           │                │
           └───────┬────────┘
                   ▼
             Weighted Sum
               Output (Y)
```

For an input matrix of token representations $X \in \mathbb{R}^{N \times d_{\text{in}}}$, we compute linear projections:

$$Q = X W^Q, \quad K = X W^K, \quad V = X W^V$$

The attention scores are calculated using the scaled dot-product kernel:

$$A = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_{QK}}}\right)$$

$$\text{Output } Y = A V$$

Where:
* $Q \in \mathbb{R}^{N_Q \times d_{QK}}$ represents the **Queries** (what current tokens are looking for).
* $K \in \mathbb{R}^{N_{KV} \times d_{QK}}$ represents the **Keys** (what previous tokens offer).
* $V \in \mathbb{R}^{N_{KV} \times d_V}$ represents the **Values** (the actual semantic information returned).

### Prefill Phase vs. Decode Phase

LLM inference consists of two distinct operational phases:

```
1. PREFILL PHASE (Prompt Processing)
   Input: "Host ETF-04 memory usage 94%" (N tokens simultaneously)
   Operation: Compute Q, K, V for all N tokens in parallel.
   Bottleneck: COMPUTE-BOUND (Matrix Multiplication Matrix-Matrix / GEMM)

2. DECODE PHASE (Autoregressive Generation)
   Input: Generated token t_i -> Predict token t_{i+1}
   Operation: Compute Q for 1 token; fetch K, V of all past tokens from cache.
   Bottleneck: MEMORY-BANDWIDTH-BOUND (Matrix-Vector / GEMM-GEMV)
```

### The KV Cache: Eliminating $O(N^3)$ Redundant Compute

During autoregressive decoding, to generate token $t_{N+1}$, the model requires the Key ($K$) and Value ($V$) representations of all preceding tokens $t_1, t_2, \dots, t_N$.

* **Without KV Caching:** At step $N$, all historical $K$ and $V$ matrices across all layers must be recomputed from scratch. The cumulative compute complexity scales quadratically per sequence, resulting in an aggregate generation complexity of **$O(N^3 \cdot d)$**.
* **With KV Caching:** At step $N$, we compute $K$ and $V$ *only for the newly generated token* and append them to an in-memory cache tensor. This reduces the per-token decode complexity to **$O(N \cdot d)$** and total generation complexity to **$O(N^2 \cdot d)$**.

#### Exact KV Cache VRAM Formula

The VRAM required to store the KV cache for a single active sequence of length $L$ is:

$$\text{Memory}_{\text{KV}} = 2 \times N_{\text{layers}} \times N_{\text{KV-heads}} \times d_{\text{head}} \times L \times \text{BytesPerElement}$$

For **Qwen 3.6-35B** ($N_{\text{layers}} = 64$, $N_{\text{KV-heads}} = 8$, $d_{\text{head}} = 128$) at precision FP16 (2 bytes):

$$\text{Memory}_{\text{KV/token}} = 2 \times 64 \times 8 \times 128 \times 2 = 262,144 \text{ bytes} \approx 256 \text{ KB / token}$$

For a full `131072` context length, a single sequence's KV cache requires:

$$131,072 \times 256 \text{ KB} = 33,554,432 \text{ KB} \approx 33.55 \text{ GB VRAM}$$

Enabling **FP8 KV Cache** ($1 \text{ byte / element}$) instantly cuts this footprint in half to **~16.77 GB VRAM** per full-length sequence.

### PagedAttention: Virtual Memory for KV Caches

Standard PyTorch allocations require contiguous CUDA memory blocks for KV tensors. Because request lengths are unpredictable, naive systems pre-allocate contiguous space for `max_model_len` (128k tokens), leading to severe **external and internal memory fragmentation** (up to 60-80% wasted VRAM).

vLLM's **PagedAttention** solves this by adapting the operating system concept of **Virtual Memory with Paging**:

```
Logical KV Cache (Sequence Space)
┌──────────┬──────────┬──────────┬──────────┐
│ Block 0  │ Block 1  │ Block 2  │ Block 3  │  (Tokens 0..63)
└────┬─────┴────┬─────┴────┬─────┴────┬─────┘
     │          │          │          │
     ▼          ▼          ▼          ▼
Block Table (Virtual Translation Map)
┌──────────┬──────────┬──────────┬──────────┐
│ Page #12 │ Page #45 │ Page #07 │ Page #89 │
└────┬─────┴────┬─────┴────┬─────┴────┬─────┘
     │          │          │          │
     ▼          ▼          ▼          ▼
Physical GPU Memory Pools (Non-Contiguous RAM Pages)
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Physical     │  │ Physical     │  │ Physical     │
│ Block #07    │  │ Block #12    │  │ Block #45    │
└──────────────┘  └──────────────┘  └──────────────┘
```

* Memory is divided into small fixed-size physical blocks (e.g., 16 or 32 tokens per block).
* Blocks are allocated dynamically on-demand as tokens are generated.
* Virtual block tables map logical sequence positions to physical GPU memory addresses, eliminating memory fragmentation and allowing utilization above **96%**.

### Prefix Caching: Accelerating Agentic Tool Calling

In our ETF Lab agent setup, every prompt sent to Qwen includes a massive static prefix:
1. System persona and monitoring policy.
2. Complete JSON Schemas for available remediation tools (`enable-auto-tool-choice`).
3. Historical host configuration baselines.

Without prefix caching, vLLM must re-run the compute-heavy Prefill Phase over these thousands of static prompt tokens on **every single request**.

```
              Radix Tree Prefix Cache Structure
              
                     [Root: System Prompt]
                      (Tokens 0..1023)
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   [Tool Schema Set A]               [Tool Schema Set B]
    (Tokens 1024..2047)               (Tokens 1024..2047)
            │                                 │
            ▼                                 ▼
[User Request: Host #1]           [User Request: Host #2]
 (Tokens 2048..2100)               (Tokens 2048..2115)
```

vLLM's **Automatic Prefix Caching (APC)** organizes KV cache blocks in a **Radix Tree**:
* When a new prompt arrives, vLLM hashes the token blocks.
* If the prefix tokens match an existing branch in the Radix tree, vLLM **skips prefill entirely** for those tokens and reuses the cached $K, V$ blocks.
* Latency to first token (TTFT) for agentic requests drops from several seconds to **milliseconds**.

---

## 4. Production Optimization & Tuning Blueprint

Equipped with these architectural insights, we establish a concrete tuning blueprint for our RTX 6000 Blackwell server.

### VRAM Allocation Map (RTX 6000 ~97 GB)

Below is the recommended physical memory distribution balancing vLLM, TEI, and system overhead:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       RTX 6000 Total Memory: ~97,280 MB                         │
├──────────────────────────────┬───────────────────────────────┬──────────────────┤
│ Component                    │ Reserved VRAM                 │ Percentage       │
├──────────────────────────────┼───────────────────────────────┼──────────────────┤
│ CUDA Driver & MPS Context    │ ~4,000 MB                     │ ~4.1%            │
│ TEI Service (Embedding Model)│ ~8,192 MB                     │ ~8.4%            │
│ vLLM Engine Target (0.82)    │ ~79,770 MB                    │ ~82.0%           │
│   ├─ Model Weights (Qwen FP8)│   ~35,000 MB                  │   (36.0%)        │
│   └─ Paged KV Cache Pool     │   ~44,770 MB                  │   (46.0%)        │
│ Unallocated Safety Buffer    │ ~5,318 MB                     │ ~5.5%            │
└──────────────────────────────┴───────────────────────────────┴──────────────────┘
```

### Actionable vLLM Parameter Tuning

To fix underutilization while protecting co-located services, update the vLLM deployment flags as follows:

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.6-35B-A3B-FP8 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.82 \
  --max-model-len 65536 \
  --max-num-seqs 256 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen_tool_parser
```

#### Rationale for Parameter Changes

1. **`gpu-memory-utilization` (`0.60` $\rightarrow$ `0.82`):**
   * Increases vLLM's memory allocation from **58.3 GB to 79.8 GB**.
   * Expands the available KV cache pool from **~23 GB to ~44.7 GB** (a **94% increase** in cache capacity).
   * Safely leaves **~17.5 GB VRAM** for MPS TEI (~8 GB), CUDA overhead (~4 GB), and safety headroom (~5.5 GB).
2. **`kv-cache-dtype` (`auto` $\rightarrow$ `fp8`):**
   * Uses FP8 (E5M2) quantization for the KV cache.
   * Doubles the token capacity of the KV cache pool, effectively allowing over **174,000 cumulative cached tokens** in memory simultaneously.
3. **`max-model-len` (`131072` $\rightarrow$ `65536`):**
   * Caps individual sequence pre-allocations to 64k tokens (more than sufficient for agent telemetry logs).
   * Prevents an anomalous edge-case query from monopolizing the KV cache table.

### Model Migration Decision Matrix: Qwen 3.6-35B vs. Qwen 3.8-27B

Should the ETF Lab migrate to **Qwen 3.8-27B**?

```
                                  Migration Decision Tree
                                             │
                       Is low latency for single requests critical?
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
                     YES                                            NO
                      │                                             │
      Migrate to Qwen 3.8-27B FP4                       Keep Qwen 3.6-35B FP8
  - VRAM Weight Footprint: ~13.5 GB               - High Prefix Cache hit rate masks prefill
  - Leaves ~66 GB VRAM for KV Cache               - Active ~3B params give fast decode latency
  - Allows co-locating a Draft Model              - Excellent tool-calling stability
```

* **Keep Qwen 3.6-35B-A3B (FP8)** if tool-calling fidelity and low decode latency per token are paramount. The MoE structure executes only ~3B parameters per token during decode, giving fast performance while Automatic Prefix Caching eliminates prefill latency.
* **Migrate to Qwen 3.8-27B (FP4)** if you wish to deploy **Speculative Decoding**. In FP4, Qwen 3.8-27B occupies only ~13.5 GB VRAM, allowing you to run a small draft model (e.g., Qwen 1.5B) on the same GPU to speculative-decode tokens, dramatically increasing throughput for bursty workloads.

---

## 5. Summary & Checklist for Engineers

- [x] **Audit VRAM Boundaries:** Ensure co-located MPS services (TEI) have explicit memory ceilings set via `CUDA_MPS_PINNED_STATIC_MEMING`.
- [x] **Maximize Prefix Caching:** Keep system prompts and tool JSON schemas strictly deterministic across agent invocations to maintain a >90% Radix tree cache hit rate.
- [x] **Enable FP8 KV Cache:** Set `--kv-cache-dtype fp8` to halve memory overhead per token with minimal loss in precision.
- [x] **Tune vLLM Ratio:** Elevate `gpu-memory-utilization` to `0.82` on 97 GB hardware to expand KV cache capacity while avoiding OOM crashes with MPS TEI.
