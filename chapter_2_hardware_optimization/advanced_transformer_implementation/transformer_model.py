import torch
import torch.nn as nn
import numpy as np
from typing import Optional

class LanguageModel(object):
    def get_next_char_log_probs(self, context: str) -> np.ndarray:
        raise Exception("Only implemented in subclasses")

    def get_log_prob_sequence(self, next_chars: str, context: str) -> float:
        raise Exception("Only implemented in subclasses")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, seq_len]
        return x + self.embedding(positions)


class FlashSelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.05):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        # x: [B, L, D]
        B, L, D = x.shape
        qkv = self.qkv_proj(x)  # [B, L, 3*D]
        
        # Reshape to [B, L, 3, nhead, head_dim] and permute to [3, B, nhead, L, head_dim]
        qkv = qkv.reshape(B, L, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each is [B, nhead, L, head_dim]
        
        # Native PyTorch Scaled Dot Product Attention
        # Automatically selects FlashAttention or Memory-Efficient Attention on GPU
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal and L > 1
        )  # [B, nhead, L, head_dim]
        
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.05):
        super().__init__()
        self.attn = FlashSelfAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN residual structure
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, max_len: int = 1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # src: [B, L]
        is_batched = len(src.shape) == 2
        if not is_batched:
            src = src.unsqueeze(0)
            
        x = self.embedding(src) * np.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        logits = self.output_layer(x)
        log_probs = torch.log_softmax(logits, dim=-1)
        
        if not is_batched:
            log_probs = log_probs.squeeze(0)
            
        return log_probs


class NeuralLanguageModel(LanguageModel):
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
            
            # Predict
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
