# Advanced Optimization Strategies: Beyond Standard Training

To scale LLMs and build low-latency production pipelines, standard training optimizations are not enough. We must employ advanced runtime architectures, IO-aware algorithms, and hardware-specific compilation techniques. 

This document summarizes the core advanced optimization strategies available today, how they resolve performance bottlenecks, and their primary use cases.

---

## 1. Summary Matrix

| Strategy | Primary Bottleneck Solved | Core Mechanism | Optimal Use Case |
|---|---|---|---|
| **Quantization** | VRAM Capacity & VRAM Bandwidth | Reduces weight/activation numerical precision (e.g., FP16 $\to$ FP8/INT4) | Large model serving, memory-constrained consumer GPUs |
| **FlashAttention** | Attention VRAM IO Bottleneck | Blocks computation using SRAM to avoid writing $O(N^2)$ attention matrices | Long-context training, training prefill, prompt processing |
| **PagedAttention** | VRAM Memory Fragmentation (KV Cache) | OS-style page tables for dynamic, non-contiguous KV cache allocation | High-concurrency serving (e.g. vLLM) |
| **SGLang** | Prefill Compute & Prefix Overhead | Radix Tree prefix caching and structured parsing compiler | Multi-turn agent systems, structured JSON output generation |
| **TensorRT-LLM** | NVIDIA Hardware Operator Latencies | Fused graph compilation and arch-specific kernels | Production-grade LLM inference at scale on NVIDIA GPUs |

---

## 2. Deep Dive of Strategies

### A. Quantization (FP8, INT8, INT4, NF4)
* **What it does**: Converts the high-precision floating-point values of weights, activations, or the Key-Value (KV) cache into lower bit-width formats.
* **How it reduces bottlenecks**:
  * **Memory Bandwidth Bound**: Autoregressive decoding is bottlenecked by the time it takes to stream model weights from VRAM to GPU registers. Quantizing to 4-bit cuts VRAM memory transactions by $4\times$, speeding up token generation proportionally.
  * **Memory Capacity Bound**: Fits a large model (e.g., Llama-3 70B, which requires 140 GB in FP16) into smaller hardware limits (e.g., ~40 GB in INT4, fitting on a single RTX A6000 or dual RTX 3090s).
* **When to apply**:
  * During inference, especially in memory-bandwidth constrained situations.
  * When running large LLMs on edge or consumer hardware.
  * Fine-tuning with limited VRAM (e.g., QLoRA).

### B. FlashAttention (FlashAttention 1, 2, & 3)
* **What it does**: A hardware-aware algorithm that calculates exact attention. Standard attention materializes the intermediate $N \times N$ attention score matrix in VRAM ($O(N^2)$ reads/writes). FlashAttention tiles the calculation into blocks, running softmax scaling and GEMMs entirely on fast, on-chip SRAM.
* **How it reduces bottlenecks**:
  * **VRAM IO Bound**: Eliminates massive VRAM read/write cycles for attention matrices. This reduces global memory traffic from quadratic $O(N^2)$ to linear $O(N)$ with respect to sequence length.
  * **Arithmetic Intensity**: Turns a heavily memory-bound block into a highly parallel compute-bound operation, letting cores run at maximum speed.
* **When to apply**:
  * Long sequence training and context lengths ($N \ge 1024$).
  * Prompt processing (prefill phase) of LLM generation.
  * Modern NVIDIA GPUs (Ampere, Ada, Hopper) supporting tensor core instructions.

### C. PagedAttention
* **What it does**: A virtual memory allocation strategy (used in engines like vLLM) that manages the KV cache. Instead of allocating large, contiguous buffers based on maximum sequence length, it breaks the KV cache into fixed-size blocks (pages) and maps them dynamically via a page table.
* **How it reduces bottlenecks**:
  * **VRAM Waste (Capacity)**: Standard contiguous allocation wastes up to $60\% - 80\%$ of VRAM space due to reservation for maximum lengths and internal/external fragmentation. PagedAttention cuts memory waste to under $4\%$.
  * **Concurrency Limit**: By freeing VRAM capacity, PagedAttention allows batch sizes to scale $2\times$ to $4\times$ larger during serving, drastically increasing server throughput.
* **When to apply**:
  * Enterprise model serving and web APIs.
  * High-concurrency, multi-user inference serving.

### D. SGLang (Structured Generation Language)
* **What it does**: A specialized execution runtime and engine optimized for structured prompting, multi-agent frameworks, and schema-constrained generations (like JSON validation).
* **How it reduces bottlenecks**:
  * **RadixAttention (Prefix Caching)**: Automatically caches KV cache pages across different generation calls using a Radix Tree. If multiple prompts share a common system prompt, prefix, or history, SGLang reuses the existing KV cache, bypassing prompt compilation/prefill completely.
  * **Structured State Machines**: Interleaves parsing logic directly with generation to prevent the GPU from waiting for CPU parsing feedback.
* **When to apply**:
  * Multi-turn chat applications, LLM agents, and pipelines with shared system prompts.
  * Extracting JSON data or structured syntax from LLMs.

### E. TensorRT-LLM
* **What it does**: NVIDIA's compiled inference engine framework. It takes PyTorch models, optimizes the computation graphs, fuses adjacent operators (e.g., GEMM + bias + activation), and generates target-specific execution plans.
* **How it reduces bottlenecks**:
  * **In-Flight Batching**: Dynamically schedules incoming requests at the token level, preventing fast generations from getting stuck behind slow ones in static batch boundaries.
  * **NVIDIA Hardware Optimization**: Automatically runs custom Assembly-level CUDA kernels tuned specifically for the GPU's memory bus width, cache sizes, and SM count.
* **When to apply**:
  * Scaling production pipelines on dedicated NVIDIA hardware environments where latency and throughput are the absolute highest priorities.
