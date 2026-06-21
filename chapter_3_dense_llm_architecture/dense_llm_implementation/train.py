import sys
import os
import random
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from chapter_3_dense_llm_architecture.dense_llm_implementation.dataset import ShakespeareDataset
from chapter_3_dense_llm_architecture.dense_llm_implementation.model import (
    DenseLLM,
    NeuralLanguageModel
)

# Device Configuration
if torch.cuda.is_available():
    print("CUDA available, using GPU")
    device = torch.device("cuda")
else:
    print("CUDA not available, using CPU")
    device = torch.device("cpu")


class CharacterTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}

    class Encoding:
        def __init__(self, ids):
            self.ids = ids

    def encode(self, text: str):
        ids = [self.char_to_idx[c] for c in text if c in self.char_to_idx]
        return self.Encoding(ids)

    def decode(self, ids: list) -> str:
        return "".join([self.idx_to_char[i] for i in ids])

    def __len__(self) -> int:
        return self.vocab_size


def estimate_batch_flops(batch_size: int, seq_len: int, num_params: int, num_layers: int, d_model: int) -> float:
    """
    Analytically estimate FLOPs per training step.
    Formula: 6 * P * tokens + 12 * layers * seq_len * d_model * tokens
    """
    total_tokens = batch_size * seq_len
    flops_per_token = 6 * num_params + 12 * num_layers * seq_len * d_model
    return total_tokens * flops_per_token


def estimate_step_memory_traffic(num_params: int, batch_size: int, seq_len: int, num_layers: int, d_model: int, dim_feedforward: int, nhead: int, nhead_kv: int) -> float:
    """
    Analytically estimate VRAM memory traffic (bytes read/written) per training step under FP16 AMP.
    Account for Grouped-Query Attention (GQA) and SwiGLU layers.
    """
    # 1. Parameter traffic under FP16 AMP (Weights read, Optimizer reads/writes, gradients write):
    param_traffic = 32.0 * num_params

    # 2. Activation traffic (FW write + BW read = 2 * Activation size):
    # Embeddings output: B * L * d_model * 2 bytes
    emb_act = batch_size * seq_len * d_model * 2.0

    # Layer activations (FP16):
    layer_act = 0.0
    for _ in range(num_layers):
        # Self-attention input: B * L * d_model * 2 bytes
        attn_in = batch_size * seq_len * d_model * 2.0
        
        # QKV projection outputs (accounting for GQA projection dimensions):
        # Q: B * L * d_model * 2 bytes
        # K, V: B * L * (d_model * nhead_kv / nhead) * 2 bytes each
        q_act = batch_size * seq_len * d_model * 2.0
        kv_act = 2.0 * batch_size * seq_len * d_model * (nhead_kv / nhead) * 2.0
        
        # Attention scores matrix (before Softmax): B * nhead * L * L * 2 bytes
        attn_matrix = batch_size * nhead * seq_len * seq_len * 2.0
        
        # Attention output projection input: B * L * d_model * 2 bytes
        attn_out = batch_size * seq_len * d_model * 2.0
        
        # SwiGLU input & intermediate gate projections:
        # Input to MLP: B * L * d_model * 2 bytes
        # Intermediate outputs (gate + up projection): 2 * B * L * dim_feedforward * 2 bytes
        mlp_in = batch_size * seq_len * d_model * 2.0
        swiglu_gate_up = 2.0 * batch_size * seq_len * dim_feedforward * 2.0
        
        # SwiGLU down projection input: B * L * dim_feedforward * 2.0
        swiglu_down = batch_size * seq_len * dim_feedforward * 2.0
        
        layer_act += (attn_in + q_act + kv_act + attn_matrix + attn_out + mlp_in + swiglu_gate_up + swiglu_down)

    total_activations = emb_act + layer_act
    activation_traffic = 2.0 * total_activations

    # 3. Input & Label traffic:
    io_traffic = batch_size * seq_len * 4.0 + batch_size * seq_len * 8.0

    return param_traffic + activation_traffic + io_traffic


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device, vocab_size: int):
    model.eval()
    loss_fn = nn.NLLLoss(reduction='sum')
    total_loss, total_tokens, total_correct = 0.0, 0, 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            log_probs = model(inputs)  # [B, L, V]
            loss = loss_fn(log_probs.view(-1, vocab_size), targets.view(-1))
            total_loss += loss.item()
            total_tokens += targets.numel()
            preds = log_probs.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()

    ppl = np.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')
    acc = total_correct / total_tokens if total_tokens > 0 else 0.0
    return ppl, acc


def train_lm(train_dataset: ShakespeareDataset, val_dataset: ShakespeareDataset, tokenizer: CharacterTokenizer, device: torch.device):
    vocab_size = len(tokenizer)
    
    # Modern Dense LLM Hyperparameters (Chapter 3)
    d_model = 512
    nhead = 8
    nhead_kv = 2  # Grouped-Query Attention (group size = 4)
    num_layers = 6
    dim_feedforward = 2048
    chunk_size = train_dataset.block_size  # 256
    batch_size = 256
    num_epochs = 3
    learning_rate = 1e-3
    
    # RTX 4070 Super Peak Theoretical FP16 Tensor Performance = 142.2 TFLOPs/sec
    PEAK_TFLOPS = 142.2

    print(f"Training parameters: d_model={d_model}, nhead={nhead}, nhead_kv={nhead_kv}, num_layers={num_layers}, vocab_size={vocab_size}")

    model = DenseLLM(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        nhead_kv=nhead_kv,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        max_len=1024
    ).to(device)
    
    # Count model parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total Model Parameters (P): {num_params:,}")

    # Compile model using PyTorch 2.0
    print("Compiling model via torch.compile...")
    model = torch.compile(model)
    print("Model compilation wrapper initialized.")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.NLLLoss()
    
    # GradScaler for AMP
    scaler = torch.cuda.amp.GradScaler()

    # DataLoaders with prefetching and pinning
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "val_ppl": [],
        "val_acc": [],
        "epoch_time_seconds": [],
        "throughput_tokens_sec": [],
        "goodput_tokens_sec": [],
        "peak_vram_allocated_mb": [],
        "peak_vram_reserved_mb": [],
        "tflops_per_sec": [],
        "mfu_percent": [],
        "memory_bandwidth_gb_sec": [],
        "arithmetic_intensity": []
    }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss, num_batches = 0.0, 0
        
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            
        epoch_start_time = time.perf_counter()

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Mixed-precision context
            with torch.cuda.amp.autocast():
                log_probs = model(inputs)
                loss = loss_fn(log_probs.view(-1, vocab_size), targets.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            num_batches += 1

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start_time

        # Metrics calculations
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        dev_ppl, dev_acc = evaluate(model, val_loader, device, vocab_size)

        avg_step_time = epoch_time / num_batches if num_batches > 0 else 1.0
        flops_per_step = estimate_batch_flops(batch_size, chunk_size, num_params, num_layers, d_model)
        
        total_tokens_step = batch_size * chunk_size
        avg_step_throughput = (total_tokens_step * num_batches) / epoch_time if epoch_time > 0 else 0.0
        avg_step_goodput = avg_step_throughput

        tflops_sec = (flops_per_step / avg_step_time) / 1e12 if avg_step_time > 0 else 0.0
        mfu = (tflops_sec / PEAK_TFLOPS) * 100

        step_memory_traffic = estimate_step_memory_traffic(
            num_params=num_params,
            batch_size=batch_size,
            seq_len=chunk_size,
            num_layers=num_layers,
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            nhead_kv=nhead_kv
        )
        memory_bandwidth_gb_sec = (step_memory_traffic / avg_step_time) / 1e9 if avg_step_time > 0 else 0.0
        arithmetic_intensity = flops_per_step / step_memory_traffic if step_memory_traffic > 0 else 0.0

        peak_vram_allocated = 0.0
        peak_vram_reserved = 0.0
        if device.type == "cuda":
            peak_vram_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            peak_vram_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

        # Log metrics
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_loss)
        history["val_ppl"].append(dev_ppl)
        history["val_acc"].append(dev_acc)
        history["epoch_time_seconds"].append(epoch_time)
        history["throughput_tokens_sec"].append(avg_step_throughput)
        history["goodput_tokens_sec"].append(avg_step_goodput)
        history["peak_vram_allocated_mb"].append(peak_vram_allocated)
        history["peak_vram_reserved_mb"].append(peak_vram_reserved)
        history["tflops_per_sec"].append(tflops_sec)
        history["mfu_percent"].append(mfu)
        history["memory_bandwidth_gb_sec"].append(memory_bandwidth_gb_sec)
        history["arithmetic_intensity"].append(arithmetic_intensity)

        print(f"Epoch {epoch+1:02d}/{num_epochs:02d} | "
              f"Train Loss: {avg_loss:.4f} | "
              f"Dev PPL: {dev_ppl:.3f} | "
              f"Dev Acc: {dev_acc:.3%} | "
              f"TFLOPs: {tflops_sec:.4f} | "
              f"MFU: {mfu:.3f}% | "
              f"VRAM: {peak_vram_allocated:.1f}MB | "
              f"Bandwidth: {memory_bandwidth_gb_sec:.4f} GB/s | "
              f"Intensity: {arithmetic_intensity:.4f} FLOPs/B")

        # Save statistics to metrics_dense.json
        metrics_path = os.path.join(os.path.dirname(__file__), "metrics_dense.json")
        with open(metrics_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Saved metrics to {metrics_path}")

    return NeuralLanguageModel(model, tokenizer, device, chunk_size)


if __name__ == "__main__":
    # Load Tiny Shakespeare text
    dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.abspath(os.path.join(
        dir_path, "..", "..", 
        "chapter_1_transformers", "basic_transformer_implementation", 
        "data", "tiny_shakespeare.txt"
    ))
    
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    print("Initializing CharacterTokenizer...")
    tokenizer = CharacterTokenizer(text)

    print("Loading Train/Val Datasets...")
    chunk_size = 256  # Sequence length 256
    train_dataset = ShakespeareDataset("train", tokenizer, chunk_size)
    val_dataset = ShakespeareDataset("val", tokenizer, chunk_size)

    print("Starting Model Training (DenseLLM Chapter 3)...")
    wrapped_model = train_lm(train_dataset, val_dataset, tokenizer, device)
    print("Training finished successfully!")
