import torch
import torch.nn as nn
import numpy as np
from typing import Optional

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm).
    Normalizes input by scaling by root mean square without subtracting the mean,
    saving memory access overhead.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS along the last dimension
        variance = x.pow(2).mean(-1, keepdim=True)
        # Scale and apply learnable weight parameter gamma
        return x * torch.rsqrt(variance + self.eps) * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cosine and sine frequency tables for Rotary Position Embeddings (RoPE).
    dim is the head_dim.
    """
    assert dim % 2 == 0, "dim (head_dim) must be even for RoPE"
    # Compute base theta frequencies
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, dtype=torch.float32)  # [end]
    freqs = torch.outer(t, freqs).float()  # [end, dim // 2]
    # Replicate freqs so we can apply rotation on coordinate pairs [cos, sin]
    freqs = torch.cat([freqs, freqs], dim=-1)  # [end, dim]
    return torch.cos(freqs), torch.sin(freqs)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half of the dimensions of the input vector.
    Used to implement complex number rotation: [x1, x2] -> [-x2, x1].
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies Rotary Position Embeddings (RoPE) to Query or Key tensor x.
    x: [B, L, H, D]
    cos, sin: [L, D] (representing precomputed rotation matrices per position)
    """
    # Reshape cos/sin to [1, L, 1, D] to broadcast across batch and heads
    cos = cos[: x.shape[1]].unsqueeze(0).unsqueeze(2)
    sin = sin[: x.shape[1]].unsqueeze(0).unsqueeze(2)
    # Apply standard 2D rotation formula on coordinates
    return (x * cos) + (rotate_half(x) * sin)


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA).
    Allows sharing Key/Value heads across groups of Query heads to reduce KV Cache footprint.
    """
    def __init__(self, d_model: int, nhead: int, nhead_kv: int, dropout: float = 0.05):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.nhead_kv = nhead_kv
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        assert nhead % nhead_kv == 0, "nhead must be divisible by nhead_kv (group size must be integer)"
        self.num_groups = nhead // nhead_kv
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, nhead * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, nhead_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, nhead_kv * self.head_dim, bias=False)
        self.out_proj = nn.Linear(nhead * self.head_dim, d_model, bias=False)
        
        self.dropout = dropout

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        B, L, D = x.shape
        
        # Project and reshape outputs to [B, L, H, head_dim]
        q = self.q_proj(x).view(B, L, self.nhead, self.head_dim)
        k = self.k_proj(x).view(B, L, self.nhead_kv, self.head_dim)
        v = self.v_proj(x).view(B, L, self.nhead_kv, self.head_dim)
        
        # Apply Rotary Position Embeddings
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        
        # Transpose to [B, H, L, head_dim] for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Grouped-Query repetition of Keys and Values
        if self.num_groups > 1:
            # Broadcast keys and values from nhead_kv to nhead
            k = k.unsqueeze(2).expand(B, self.nhead_kv, self.num_groups, L, self.head_dim).reshape(B, self.nhead, L, self.head_dim)
            v = v.unsqueeze(2).expand(B, self.nhead_kv, self.num_groups, L, self.head_dim).reshape(B, self.nhead, L, self.head_dim)
        
        # Perform Scaled Dot Product Attention (native PyTorch SDPA uses FlashAttention if possible)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal and L > 1
        )
        
        # Re-merge attention heads
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_proj(out)


class SwiGLU(nn.Module):
    """
    Swish Gated Linear Unit (SwiGLU).
    Splits feedforward paths into a Gated segment and an Up segment before Down-projecting.
    """
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.05):
        super().__init__()
        self.w_gate = nn.Linear(d_model, dim_feedforward, bias=False)
        self.w_up = nn.Linear(d_model, dim_feedforward, bias=False)
        self.w_down = nn.Linear(dim_feedforward, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU formula: Swish(x * W_gate) * (x * W_up) * W_down
        gate = torch.nn.functional.silu(self.w_gate(x))
        up = self.w_up(x)
        activated = gate * up
        return self.w_down(self.dropout(activated))


class DenseLLMBlock(nn.Module):
    """
    Single block of modern Llama-style Decoder model.
    Utilizes Pre-LN with RMSNorm, GQA, and SwiGLU.
    """
    def __init__(self, d_model: int, nhead: int, nhead_kv: int, dim_feedforward: int, dropout: float = 0.05):
        super().__init__()
        self.attn = GroupedQueryAttention(d_model, nhead, nhead_kv, dropout)
        self.mlp = SwiGLU(d_model, dim_feedforward, dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class DenseLLM(nn.Module):
    """
    Decoder-only Dense LLM architecture.
    """
    def __init__(self, vocab_size: int, d_model: int, nhead: int, nhead_kv: int, num_layers: int, dim_feedforward: int, max_len: int = 1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.nhead_kv = nhead_kv
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Precompute RoPE Cosine and Sine frequencies
        head_dim = d_model // nhead
        cos, sin = precompute_freqs_cis(head_dim, max_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        
        # Stack blocks
        self.layers = nn.ModuleList([
            DenseLLMBlock(d_model, nhead, nhead_kv, dim_feedforward)
            for _ in range(num_layers)
        ])
        
        self.norm = RMSNorm(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # Input shape: [B, L]
        is_batched = len(src.shape) == 2
        if not is_batched:
            src = src.unsqueeze(0)
            
        x = self.embedding(src)
        
        # Extract corresponding sequence context frequencies
        L = src.shape[1]
        cos_l = self.cos[:L]
        sin_l = self.sin[:L]
        
        # Stacks forward passes
        for layer in self.layers:
            x = layer(x, cos_l, sin_l)
            
        x = self.norm(x)
        logits = self.output_layer(x)
        log_probs = torch.log_softmax(logits, dim=-1)
        
        if not is_batched:
            log_probs = log_probs.squeeze(0)
            
        return log_probs


class NeuralLanguageModel(object):
    """
    Consistent evaluation wrapper matching Chapter 1 & 2 interfaces.
    """
    def __init__(self, model: nn.Module, tokenizer, device: torch.device, chunk_size: int = 256):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.chunk_size = chunk_size

    def get_next_char_log_probs(self, context: str) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            if len(context) == 0:
                context = " "
            truncated_context = context[-self.chunk_size:]
            encoded = self.tokenizer.encode(truncated_context)
            context_indices = encoded.ids
            context_tensor = torch.LongTensor(context_indices).unsqueeze(0).to(self.device)
            
            # Predict next token log probs
            log_probs = self.model(context_tensor)  # [1, seq_len, vocab_size]
            next_char_log_probs = log_probs[0, -1].cpu().numpy()
        return next_char_log_probs

    def get_log_prob_sequence(self, next_chars: str, context: str) -> float:
        self.model.eval()
        total_log_prob = 0.0
        temp_context = context
        with torch.no_grad():
            for char_to_predict in next_chars:
                log_probs_dist = self.get_next_char_log_probs(temp_context)
                char_idx = self.tokenizer.char_to_idx.get(char_to_predict, -1)
                if char_idx == -1:
                    total_log_prob += float('-inf')
                else:
                    total_log_prob += log_probs_dist[char_idx]
                temp_context += char_to_predict
        return total_log_prob


# ==============================================================================
# Unit Tests
# ==============================================================================
if __name__ == "__main__":
    print("Executing unit tests for DenseLLM components...")
    
    # Setup dummy data shapes
    B, L, D = 4, 16, 64
    nhead, nhead_kv = 4, 1  # 4 query heads, 1 KV head (GQA group_size = 4)
    vocab_size = 100
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running unit tests on device: {device}")
    
    # 1. Test RMSNorm
    print("\n[Test 1] Verifying RMSNorm...")
    x = torch.randn(B, L, D, device=device)
    norm = RMSNorm(D).to(device)
    y = norm(x)
    assert y.shape == x.shape, f"RMSNorm output shape mismatch: {y.shape}"
    # RMS value of normalized output should be close to 1.0 along normalized dim if gamma=1.0
    rms = torch.sqrt(y.pow(2).mean(-1))
    print(f"  -> Input shape: {x.shape}")
    print(f"  -> Output shape: {y.shape}")
    print(f"  -> Normalized RMS values (should be close to 1.0): {rms.mean().item():.4f}")
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3), "RMS normalization scaling error"
    print("  -> RMSNorm: SUCCESS")

    # 2. Test RoPE precomputation and rotation shape
    print("\n[Test 2] Verifying RoPE frequencies and rotation...")
    head_dim = D // nhead
    cos, sin = precompute_freqs_cis(head_dim, 128)
    cos, sin = cos.to(device), sin.to(device)
    assert cos.shape == (128, head_dim), f"RoPE cos shape mismatch: {cos.shape}"
    assert sin.shape == (128, head_dim), f"RoPE sin shape mismatch: {sin.shape}"
    
    q = torch.randn(B, L, nhead, head_dim, device=device)
    q_rot = apply_rotary_emb(q, cos, sin)
    assert q_rot.shape == q.shape, f"Rotated query shape mismatch: {q_rot.shape}"
    print(f"  -> Cos shape: {cos.shape}")
    print(f"  -> Q shape: {q.shape}")
    print(f"  -> Q_rot shape: {q_rot.shape}")
    print("  -> RoPE Embeddings: SUCCESS")

    # 3. Test GroupedQueryAttention
    print("\n[Test 3] Verifying Grouped-Query Attention (GQA)...")
    gqa = GroupedQueryAttention(D, nhead, nhead_kv).to(device)
    cos_l = cos[:L]
    sin_l = sin[:L]
    attn_out = gqa(x, cos_l, sin_l)
    assert attn_out.shape == x.shape, f"GQA output shape mismatch: {attn_out.shape}"
    print(f"  -> GQA input shape: {x.shape}")
    print(f"  -> GQA output shape: {attn_out.shape}")
    print("  -> GQA: SUCCESS")

    # 4. Test SwiGLU
    print("\n[Test 4] Verifying SwiGLU MLP...")
    swiglu = SwiGLU(D, 256).to(device)
    mlp_out = swiglu(x)
    assert mlp_out.shape == x.shape, f"SwiGLU output shape mismatch: {mlp_out.shape}"
    print(f"  -> SwiGLU inputs: {x.shape}")
    print(f"  -> SwiGLU outputs: {mlp_out.shape}")
    print("  -> SwiGLU: SUCCESS")

    # 5. Test Full Model
    print("\n[Test 5] Verifying Full DenseLLM Model...")
    model = DenseLLM(
        vocab_size=vocab_size,
        d_model=D,
        nhead=nhead,
        nhead_kv=nhead_kv,
        num_layers=2,
        dim_feedforward=256,
        max_len=128
    ).to(device)
    
    src = torch.randint(0, vocab_size, (B, L), device=device)
    log_probs = model(src)
    assert log_probs.shape == (B, L, vocab_size), f"DenseLLM output shape mismatch: {log_probs.shape}"
    
    # Verify backward pass / gradient flow
    loss = log_probs.sum()
    loss.backward()
    
    # Check if gradients flow to input embedding layer
    assert model.embedding.weight.grad is not None, "Gradients failed to propagate to embeddings"
    print(f"  -> DenseLLM inputs: {src.shape}")
    print(f"  -> DenseLLM outputs (Log Probs): {log_probs.shape}")
    print(f"  -> Gradients successfully propagated to embeddings: shape {model.embedding.weight.grad.shape}")
    print("  -> Full DenseLLM: SUCCESS")
    
    print("\nAll unit tests passed successfully!")
