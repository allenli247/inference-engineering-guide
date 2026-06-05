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

# Add project root to sys.path to make absolute imports work from anywhere
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from chapter_1_transformers.basic_transformer_implementation.dataset import ShakespeareDataset
from chapter_1_transformers.basic_transformer_implementation.transformer_model import (
    TransformerLM,
    NeuralLanguageModel,
    create_causal_mask
)

# Device Configuration
if torch.cuda.is_available():
    print("CUDA available, using GPU")
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    print("MPS available, using GPU")
    device = torch.device("mps")
else:
    print("CUDA/MPS not available, using CPU")
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


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device, vocab_size: int):
    """
    Evaluate perplexity and accuracy on dev/validation set using the data loader
    """
    model.eval()
    loss_fn = nn.NLLLoss(reduction='sum')
    total_loss, total_tokens, total_correct = 0.0, 0, 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            seq_len = inputs.size(1)
            causal_mask = create_causal_mask(seq_len, device)
            
            # forward pass
            log_probs = model(inputs, causal_mask)  # [B, L, V]
            
            # compute loss
            loss = loss_fn(log_probs.view(-1, vocab_size), targets.view(-1))
            total_loss += loss.item()
            total_tokens += targets.numel()

            # compute accuracy
            preds = log_probs.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()

    ppl = np.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')
    acc = total_correct / total_tokens if total_tokens > 0 else 0.0
    return ppl, acc


def train_lm(train_dataset: ShakespeareDataset, val_dataset: ShakespeareDataset, tokenizer: CharacterTokenizer, device: torch.device):
    """
    Train transformer LM and return wrapped NeuralLanguageModel with detailed metrics logging
    """
    vocab_size = len(tokenizer)
    d_model = 128
    nhead = 4
    num_layers = 2
    dim_feedforward = 256
    chunk_size = train_dataset.block_size
    batch_size = 64
    num_epochs = 10
    learning_rate = 1e-3
    
    # RTX 4070 Peak Theoretical FP16 Tensor Performance = 121.3 TFLOPs/sec
    PEAK_TFLOPS = 121.3

    print(f"Training parameters: d_model={d_model}, nhead={nhead}, num_layers={num_layers}, vocab_size={vocab_size}")

    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        max_len=1024
    ).to(device)
    
    # Count model parameters for FLOPs calculation
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total Model Parameters (P): {num_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.NLLLoss()

    # Create PyTorch DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Dictionary to store training statistics
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
        "mfu_percent": []
    }

    # Reset peak memory stats before training
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss, num_batches = 0.0, 0
        
        # Metrics to accumulate over steps
        epoch_step_times = []
        epoch_step_flops = []
        epoch_step_throughput_tokens = []
        epoch_step_goodput_tokens = []

        # Reset peak VRAM tracking per epoch
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            
        epoch_start_time = time.perf_counter()

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            seq_len = inputs.size(1)
            causal_mask = create_causal_mask(seq_len, device)

            # Synchronize to get accurate time profiles on GPUs
            if device.type == "cuda":
                torch.cuda.synchronize()
            step_start = time.perf_counter()

            # forward pass
            log_probs = model(inputs, causal_mask)  # [B, L, V]
            loss = loss_fn(log_probs.view(-1, vocab_size), targets.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize()
            step_time = time.perf_counter() - step_start

            # Calculate metrics for this step
            total_tokens = inputs.numel()  # B * L
            # For character-level model without padding, all tokens are useful (-100 is default ignore_index)
            useful_tokens = (targets != -100).sum().item()

            flops = estimate_batch_flops(batch_size, seq_len, num_params, num_layers, d_model)
            
            # Append metrics
            epoch_step_times.append(step_time)
            epoch_step_flops.append(flops)
            epoch_step_throughput_tokens.append(total_tokens)
            epoch_step_goodput_tokens.append(useful_tokens)

            total_loss += loss.item()
            num_batches += 1

        epoch_time = time.perf_counter() - epoch_start_time

        # Calculate epoch averages
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        dev_ppl, dev_acc = evaluate(model, val_loader, device, vocab_size)

        # Average step metrics
        avg_step_time = np.mean(epoch_step_times) if epoch_step_times else 1.0
        avg_step_flops = np.mean(epoch_step_flops) if epoch_step_flops else 0.0
        avg_step_throughput = np.sum(epoch_step_throughput_tokens) / epoch_time if epoch_time > 0 else 0.0
        avg_step_goodput = np.sum(epoch_step_goodput_tokens) / epoch_time if epoch_time > 0 else 0.0

        # Calculate TFLOPs/sec and MFU
        # FLOPs / step_time -> convert to TFLOPs by dividing by 10^12
        tflops_sec = (avg_step_flops / avg_step_time) / 1e12 if avg_step_time > 0 else 0.0
        mfu = (tflops_sec / PEAK_TFLOPS) * 100

        # VRAM stats
        peak_vram_allocated = 0.0
        peak_vram_reserved = 0.0
        if device.type == "cuda":
            peak_vram_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            peak_vram_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

        # Log to history dict
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

        print(f"Epoch {epoch+1:02d}/{num_epochs:02d} | "
              f"Train Loss: {avg_loss:.4f} | "
              f"Dev PPL: {dev_ppl:.3f} | "
              f"Dev Acc: {dev_acc:.3%} | "
              f"TFLOPs: {tflops_sec:.4f} | "
              f"MFU: {mfu:.3f}% | "
              f"VRAM: {peak_vram_allocated:.1f}MB")

        # Save statistics to a version-controlled JSON file at the end of each epoch
        metrics_path = os.path.join(os.path.dirname(__file__), "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Saved metrics to {metrics_path}")

    model.eval()
    return NeuralLanguageModel(model, tokenizer, device, chunk_size)


if __name__ == "__main__":
    # Load dataset text file to build character tokenizer
    data_path = os.path.join(os.path.dirname(__file__), "data", "tiny_shakespeare.txt")
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    print("Initializing CharacterTokenizer...")
    tokenizer = CharacterTokenizer(text)

    print("Loading Train/Val Datasets...")
    chunk_size = 20
    train_dataset = ShakespeareDataset("train", tokenizer, chunk_size)
    val_dataset = ShakespeareDataset("val", tokenizer, chunk_size)

    print("Starting Model Training...")
    wrapped_model = train_lm(train_dataset, val_dataset, tokenizer, device)
    print("Training finished successfully!")
