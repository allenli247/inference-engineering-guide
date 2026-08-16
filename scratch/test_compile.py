import torch
from chapter_3_dense_llm_architecture.dense_llm_implementation.model import DenseLLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenseLLM(
    vocab_size=100,
    d_model=128,
    nhead=4,
    nhead_kv=1,
    num_layers=2,
    dim_feedforward=256,
    max_len=128
).to(device)

compiled_model = torch.compile(model)
src = torch.randint(0, 100, (4, 16), device=device)

try:
    out = compiled_model(src)
    loss = out.sum()
    loss.backward()
    print("Compilation and forward/backward successful!")
except Exception as e:
    print("Error during execution:")
    import traceback
    traceback.print_exc()
