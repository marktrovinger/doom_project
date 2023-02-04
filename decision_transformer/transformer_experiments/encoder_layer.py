import torch.nn as nn
from .mha import MultiHeadAttention

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.3) -> None:
        super().__init__()