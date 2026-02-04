import torch
import torch.optim as optim
import torch.nn as nn
import sys

print(f"Python version: {sys.version}")
print(f"Torch version: {torch.__version__}")

model = nn.Linear(10, 10)
try:
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("Success")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
