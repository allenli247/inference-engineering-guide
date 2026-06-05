# Hardware Utilization Probing: Measuring Training Efficiency

When building and scaling large models, training speed is the primary bottleneck. However, "training is slow" is not an actionable complaint. To optimize training runs, we must treat the hardware (your NVIDIA RTX 4070 GPU) as a system and actively profile its utilization. This process is called **hardware utilization probing**.

In this section, we introduce the core concepts of memory and compute profiling, explain the PyTorch and CUDA APIs used to measure them, and walk through our metric-gathering workflow.

---

## 1. Core Hardware Concepts

### A. VRAM Allocation: Allocated vs. Reserved Memory
When running models on a GPU, memory is not a single contiguous pool managed directly by your Python script. Instead, PyTorch uses a **Caching Allocator** to manage VRAM.

* **Allocated Memory**: The VRAM actively holding PyTorch tensors (weights, activations, gradients, and optimizer states).
* **Reserved (Cached) Memory**: The VRAM that PyTorch's allocator has requested from the CUDA driver and is keeping in a cache. If you delete a tensor, the memory is freed from "Allocated" but remains "Reserved" so PyTorch can reuse it immediately without the heavy overhead of requesting memory from the OS again.

> [!WARNING]
> If your **Allocated Memory** exceeds your GPU's total VRAM (12 GB on the RTX 4070), you will trigger an **Out of Memory (OOM)** crash. Understanding peak allocated VRAM helps you size your maximum batch size and sequence lengths.

### B. Throughput vs. Goodput
* **Throughput**: The total number of tokens (characters, words, or subwords) processed by the hardware per second.
* **Goodput**: The rate of *useful* tokens processed per second that actually contribute to gradient updates and learning.

> [!NOTE]
> In variable-length batches, shorter sequences are padded with `<pad>` tokens to match the longest sequence in the batch. While these padding tokens are computed in the forward and backward passes (consuming FLOPs and GPU cycles), their gradients are masked out. Thus, padding tokens contribute to **Throughput** but are excluded from **Goodput**. Minimizing this gap is a key optimization.

### C. FLOPs and Model FLOPs Utilization (MFU)
* **FLOPs (Floating Point Operations)**: The total number of arithmetic operations (adds and multiplies) executed during training. We estimate this using a standard formula for the forward and backward pass:
  $$\text{FLOPs per Token} \approx 6P + 12 \times \text{layers} \times L \times d_{\text{model}}$$
  where $P$ is the parameter count, $L$ is the sequence length, and $d_{\text{model}}$ is the hidden dimension.
* **MFU (Model FLOPs Utilization)**: The ratio of achieved compute performance (TFLOPs/sec) to the hardware's peak theoretical performance. It is the gold standard for measuring GPU execution efficiency.
  * **RTX 4070 Super Peak FP16 Performance**: $\approx 142.2\text{ TFLOPs/sec}$.
    > [!NOTE]
    > **Deriving the 142.2 TFLOPs/sec Peak:**
    > Databases like TechPowerUp list the RTX 4070 Super's **Half Precision (FP16)** performance as **71.09 TFLOPS** and the **Tensor Core** performance as **284.4 TFLOPS** (sparse). 
    > - The **71.09 TFLOPS** figure represents standard vector operations on CUDA cores (non-tensor), which is twice the FP32 rate.
    > - The **284.4 TFLOPS** figure assumes **2:4 structured sparsity** (where 50% of the weights are zeroed out and skipped).
    > - Because standard neural networks (including this implementation) are **dense**, they run on the dense Tensor Core pipeline. The theoretical dense peak is exactly half of the sparse peak:
    >   $$\text{Dense Peak Tensor FP16} = \frac{284.4\text{ TFLOPS}}{2} = 142.2\text{ TFLOPS}$$
  * If your training code achieves $1.42\text{ TFLOPs/sec}$, your MFU is $\approx 1\%$. 
  * Low MFU (very common with small batch sizes or short sequence lengths) indicates that the GPU is underutilized, usually because it is waiting for CPU kernel launches (overhead-bound) or reading/writing to memory (memory-bound).

### D. Memory Bandwidth and Arithmetic Intensity
* **Memory Bandwidth**: The rate at which the GPU cores can read from or write to the high-bandwidth VRAM. For the RTX 4070 Super, this peak rate is **504 GB/s**.
* **Arithmetic Intensity**: Measured in FLOPs/Byte, this is the ratio of compute operations to memory traffic:
  $$\text{Arithmetic Intensity} = \frac{\text{Floating Point Operations (FLOPs)}}{\text{Memory Traffic (Bytes)}}$$
* **Memory Traffic**: The total number of bytes read from or written to global VRAM:
  - **Parameter & Gradient Traffic**: Each weight must be read during forward and backward passes. Gradients must be written. For FP32 parameters using the Adam optimizer, this equals $40 \times P$ bytes per step (including weights, gradients, and momentum/variance tracking states).
  - **Activation Traffic**: Intermediate activations are written during the forward pass and read during the backward pass (totaling $2 \times \text{Activation Size}$ bytes).
* **The Roofline Model**: A visualization tool mapping achieved performance (TFLOPs/sec) vs. arithmetic intensity. The "ridge point" is the boundary where the workload shifts from being **Memory-Bound** (waiting for data to transfer from VRAM) to **Compute-Bound** (waiting for math operations on the core execution units).

---

## 2. Profiling Syntax and Commands

To capture these metrics programmatically during training, we use the following APIs:

### A. Accurate Timing via CUDA Synchronization
Because GPU execution is asynchronous, Python code launches a GPU kernel and immediately returns to run the next line on the CPU while the GPU is still computing. 

If you measure time like this:
```python
# INCORRECT WAY - measures launch overhead, not execution!
start = time.perf_counter()
output = model(inputs)
duration = time.perf_counter() - start
```
You will get misleadingly fast times. To get the actual execution time, you **must synchronize** the CPU and GPU:

```python
# CORRECT WAY
if device.type == "cuda":
    torch.cuda.synchronize() # Wait for all pending GPU tasks to finish
start = time.perf_counter()

output = model(inputs)

if device.type == "cuda":
    torch.cuda.synchronize() # Wait for the current computation to complete
duration = time.perf_counter() - start
```

### B. Memory Probing
PyTorch provides memory tracking commands:
* `torch.cuda.max_memory_allocated(device)`: Returns the peak allocated memory in bytes since the last reset.
* `torch.cuda.max_memory_reserved(device)`: Returns the peak reserved memory in bytes since the last reset.
* `torch.cuda.reset_peak_memory_stats(device)`: Resets the peak trackers so you can measure VRAM usage per epoch or per step.

---

## 3. Profiling Workflow

Our metric logging workflow is implemented in [`train.py`](file:///root/inference-engineering-guide/chapter_1_transformers/basic_transformer_implementation/train.py):

1. **Parameter Count**: Calculate model size ($P$) at startup:
   ```python
   num_params = sum(p.numel() for p in model.parameters())
   ```
2. **Step-Level Tracking**: Inside the data loader loop:
   * Synchronize GPU and start a step timer.
   * Run the forward, backward, and optimization steps.
   * Synchronize GPU and stop the step timer.
   * Calculate FLOPs for the step and throughput/goodput tokens.
3. **Epoch Aggregation**: Average the step metrics, query PyTorch for peak VRAM usage, calculate MFU based on the RTX 4070 Super peak limit (142.2 TFLOPs/sec), estimate memory bandwidth and arithmetic intensity, and export these statistics to `metrics.json`.
4. **Interactive Plotting**: Load `metrics.json` in [`visualize_metrics.ipynb`](file:///root/inference-engineering-guide/chapter_1_transformers/basic_transformer_implementation/visualize_metrics.ipynb) to inspect interactive plots of compute efficiency, memory footprints, and the Roofline Model using Plotly.

---

## 4. Why Hasn't Our Training Maximized GPU Performance?

During our training runs, we achieved compute performance of **~0.5 TFLOPs/sec**, representing a Model FLOPs Utilization (MFU) of **~0.4%** relative to the RTX 4070 Super's peak dense Tensor FP16 performance (142.2 TFLOPs/sec) and **~1.4%** relative to the peak FP32 vector performance (35.5 TFLOPs/sec). 

Here is why our training pipeline is heavily bottlenecked and how these factors contribute to memory and compute underutilization:

### A. The Model is Extremely Small (Cache vs. VRAM Traffic)
Our network has only $412,737$ parameters ($\approx 1.65$ MB in FP32). 
* **Cache Fit**: Modern GPUs (like the RTX 4070 Super, which has 48 MB of L2 cache) can fit the entire set of weights, gradients, and optimizer states directly inside the fast on-chip L2 cache. 
* **Underutilized Memory Bus**: Because the parameters are cached on-chip, the GPU doesn't actually need to fetch them from VRAM. The global VRAM bandwidth (504 GB/s) is never saturated.

### B. Severe Thread Starvation (GPU Core Underutilization)
A GPU thrives on massive parallelism. To hide execution latency, it needs thousands of concurrent threads grouped into thread blocks to populate all 56 Streaming Multiprocessors (SMs).
* **Suboptimal Batching**: Our batch size of 64 and sequence length of 20 equals $1,280$ tokens per step. 
* **Tiny Matrices**: The matrix multiplications (GEMMs) in our projection layers are very small (e.g., multiplying matrices of size $1280 \times 128$ and $128 \times 128$). Small matrix multiplications generate too few thread blocks, leaving most of the GPU's CUDA cores and Tensor Cores completely idle.

### C. Domination of Memory-Bound Operations
In small models, a huge percentage of the total math operations are **element-wise** rather than large matrix multiplications:
* **Memory-Bound Kernels**: Operations like LayerNorm, Softmax, GELU, residual additions, and Dropout require reading a tensor from VRAM, performing a single simple math operation (like addition or exponentiation), and writing it back to VRAM.
* **Low Arithmetic Intensity**: These kernels have an arithmetic intensity of less than 1 FLOP/Byte. The GPU cores spend nearly 100% of their time waiting for memory transfers, resulting in low TFLOPs/sec.

### D. Host-to-Device Bottlenecks & CPU Launch Overhead
Because our GPU execution times are so short (each batch takes less than $1$ millisecond of actual GPU compute time), the pipeline is bottlenecked by the CPU:
* **Python Interpreter Latency**: Python executes code line-by-line, which takes time.
* **Kernel Launch Overhead**: The CPU takes $3$ to $10$ microseconds to launch a single GPU kernel via the CUDA driver. Since a single training step consists of dozens of small kernel launches (for embeddings, projections, attention, normalizations, and optimizer steps), the launch overhead dominates the step duration.
* **Data Transfer**: Transferring input batches from CPU RAM to GPU VRAM over the PCIe bus introduces latency for every batch, which is visible since the GPU finishes the computation almost instantly.
