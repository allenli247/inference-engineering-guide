import torch
import torch.nn as nn
import numpy as np
from typing import Optional

class LanguageModel(object):
    def get_next_char_log_probs(self, context: str) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e.
        """
        raise Exception("Only implemented in subclasses")

    def get_log_prob_sequence(self, next_chars: str, context: str) -> float:
        """
        Scores a bunch of characters following context. That is, returns:
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e.
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size: int):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context: str) -> np.ndarray:
        return np.ones([self.voc_size]) * np.log(1.0 / self.voc_size)

    def get_log_prob_sequence(self, next_chars: str, context: str) -> float:
        return np.log(1.0 / self.voc_size) * len(next_chars)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: tensor of shape [batch_size, seq_len, d_model]
        :return: x with positional encodings added
        """
        seq_len = x.shape[1]
        # Create positions on the same device as input x
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, seq_len]
        return x + self.embedding(positions)


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, max_len: int = 512):
        """
        :param vocab_size: vocabulary size
        :param d_model: embedding dimension
        :param nhead: number of attention heads
        :param num_layers: number of transformer layers
        :param dim_feedforward: dimension of feedforward network
        :param max_len: maximum sequence length for positional encoding
        """
        super().__init__()

        # vocab embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)

        # positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        # transformer encoder params
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.05,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # output layer
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        self.d_model = d_model

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param src: input tensor of shape [batch_size, seq_len]
        :param src_mask: causal mask of shape [seq_len, seq_len]
        :return: log probabilities of shape [batch_size, seq_len, vocab_size]
        """
        # handle both batched and unbatched inputs
        is_batched = len(src.shape) == 2
        if not is_batched:
            src = src.unsqueeze(0)  # Add batch dimension
        
        # embed and add positional encoding (scale embeddings by sqrt(d_model) as in paper)
        x = self.embedding(src) * np.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # transformer with causal mask
        x = self.transformer(x, mask=src_mask)
        
        # project to vocabulary
        logits = self.output_layer(x)
        
        # apply log softmax
        log_probs = torch.log_softmax(logits, dim=-1)
        
        if not is_batched:
            log_probs = log_probs.squeeze(0)  # Remove batch dimension
        
        return log_probs


class NeuralLanguageModel(LanguageModel):
    def __init__(self, model: nn.Module, tokenizer, device: torch.device, chunk_size: int = 20):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.chunk_size = chunk_size

    def get_next_char_log_probs(self, context: str) -> np.ndarray:
        """
        return log_probs for next character given a TRUNCATED context
        """
        self.model.eval()
        
        with torch.no_grad():
            # Prepend space as start token if context is empty
            if len(context) == 0:
                context = " "
            
            truncated_context = context[-self.chunk_size:]
            
            # convert the truncated context to indices
            encoded = self.tokenizer.encode(truncated_context)
            context_indices = encoded.ids
            context_tensor = torch.LongTensor(context_indices).unsqueeze(0).to(self.device)  # [1, seq_len]
            
            # create causal mask for this sequence length
            seq_len = context_tensor.shape[1]
            causal_mask = create_causal_mask(seq_len, self.device)
            
            # get predictions
            log_probs = self.model(context_tensor, causal_mask)  # [1, seq_len, vocab_size]
            
            # return log probs for the character following the final one in the context
            next_char_log_probs = log_probs[0, -1].cpu().numpy()
            
        return next_char_log_probs

    def get_log_prob_sequence(self, next_chars: str, context: str) -> float:
        """
        Scores a sequence by calling the consistent get_next_char_log_probs
        repeatedly to pass the sanity check.
        """
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


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Creates a causal mask to prevent attending to future positions
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
    return mask
