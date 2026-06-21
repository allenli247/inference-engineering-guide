import torch
import os
from torch.utils.data import Dataset

class ShakespeareDataset(Dataset):
    def __init__(self, split: str, tokenizer, block_size: int):
        super().__init__()
        self.block_size = block_size
        
        # Load the raw text file
        dir_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.abspath(os.path.join(
            dir_path, "..", "..", 
            "chapter_1_transformers", "basic_transformer_implementation", 
            "data", "tiny_shakespeare.txt"
        ))
        
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        # Tokenize the entire corpus into token IDs
        encoded = tokenizer.encode(text)
        token_ids = encoded.ids
        
        # Create Train/Val split (90% train, 10% val)
        split_idx = int(0.9 * len(token_ids))
        
        if split == "train":
            self.data = token_ids[:split_idx]
        elif split == "val":
            self.data = token_ids[split_idx:]
        else:
            raise ValueError("Split must be either 'train' or 'val'")
            
        # Convert to a PyTorch tensor
        self.data = torch.tensor(self.data, dtype=torch.long)
        
        # Calculate non-overlapping chunks (each chunk needs block_size + 1 tokens)
        self.num_chunks = len(self.data) // (block_size + 1)

    def __len__(self) -> int:
        return self.num_chunks

    def __getitem__(self, idx: int):
        start_idx = idx * (self.block_size + 1)
        x = self.data[start_idx : start_idx + self.block_size]
        y = self.data[start_idx + 1 : start_idx + self.block_size + 1]
        return x, y
