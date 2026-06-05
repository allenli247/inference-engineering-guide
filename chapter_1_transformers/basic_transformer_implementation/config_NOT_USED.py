# config.py
from dataclasses import dataclass
import torch

@dataclass
class ModelConfig:
    # Architecture
    block_size: int = 256        # Max context length (tokens). This is the window the transformer looks at to generate the next token.
    n_embd: int = 384            # Embedding dimension. The size of the vector space that the transformer operates in. Higher is better, but more computationally expensive.
    n_head: int = 6              # Number of attention heads (384 / 6 = 64 dim per head). Each head processes different aspects of the sequence.
    n_layer: int = 6             # Number of Transformer blocks. Increasing this increases the depth of the model and its ability to learn complex patterns.
    dropout: float = 0.1         # Regularization to prevent overfitting

@dataclass
class TrainingConfig:
    # Hardware & Loops
    batch_size: int = 64         # Optimized for RTX 4070 Super
    learning_rate: float = 3e-4  # Standard AdamW learning rate for Transformers
    max_iters: int = 5000        # Total training steps
    eval_interval: int = 500     # How often to check validation loss
    eval_iters: int = 200        # Number of batches to average during evaluation
    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint_path: str = "checkpoints/latest_model.pt" # Path to save the model checkpoints