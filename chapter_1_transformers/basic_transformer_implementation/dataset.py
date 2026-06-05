import torch
import os
from torch.utils.data import Dataset
from tokenizers import Tokenizer

class ShakespeareDataset(Dataset): # Dataset abstract base class requires init, len, and getitem 
    def __init__(self, split: str, tokenizer: Tokenizer, block_size: int):
        # Inheritance allows the ShakespeareDataset to inherit Dataset properties, allowing it to be recognized by DataLoader
        super().__init__()  # initialize the parent class, Dataset, before the custom setup below
        self.block_size = block_size
        
        # Load the raw text file
        dir_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(dir_path, "data", "tiny_shakespeare.txt")
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        # Tokenize the entire corpus into token IDs
        encoded = tokenizer.encode(text)  # the encoded object contains lists of token ids, the tokens, positional indices, and attention masks
        token_ids = encoded.ids   # get token IDs which correspond to the row index in the embedding matrix
        
        # Create a strict Train/Val split (90% train, 10% val)
        split_idx = int(0.9 * len(token_ids))
        
        if split == "train":
            self.data = token_ids[:split_idx]
        elif split == "val":
            self.data = token_ids[split_idx:]
        else:
            raise ValueError("Split must be either 'train' or 'val'")
            
        # Convert to a PyTorch tensor
        self.data = torch.tensor(self.data, dtype=torch.long)

    def __len__(self) -> int:
        # the last valid index we can start a window from is block size from the final index in the data
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int):
        # x is the sequence of tokens the model sees with each block
        x = self.data[idx : idx + self.block_size]
        
        # y is the exact same sequence, but shifted right by 1 token (the target labels)
        y = self.data[idx + 1 : idx + self.block_size + 1]
        
        return x, y