import torch
from src.constants import *

# Load the .pt file
loaded_data = torch.load(RolloutModelPath_10x10_4v2, map_location=torch.device('cpu'))

# Access the loaded data
print(loaded_data)