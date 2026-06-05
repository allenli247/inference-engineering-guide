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
  * **RTX 4070 Peak FP16 Performance**: $\approx 121.3\text{ TFLOPs/sec}$.
  * If your training code achieves $1.2\text{ TFLOPs/sec}$, your MFU is $\approx 1\%$. 
  * Low MFU (very common with small batch sizes or short sequence lengths) indicates that the GPU is underutilized, usually because it is waiting for CPU kernel launches (overhead-bound) or reading/writing to memory (memory-bound).

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
3. **Epoch Aggregation**: Average the step metrics, query PyTorch for peak VRAM usage, calculate MFU based on the RTX 4070 peak limit (121.3 TFLOPs/sec), and export these statistics to `metrics.json`.
4. **Interactive Plotting**: Load `metrics.json` in [`visualize_metrics.ipynb`](file:///root/inference-engineering-guide/chapter_1_transformers/basic_transformer_implementation/visualize_metrics.ipynb) to inspect interactive plots of compute efficiency and memory footprints using Plotly.
