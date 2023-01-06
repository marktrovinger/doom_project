import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=4, num_heads=2, dropout=0.3) -> None:
        super().__init__()

        self.d = d_model // num_heads
        
        self.d_model = d_model
        self.num_heads = num_heads

        self.dropout = nn.Dropout(dropout)

        # create the linear embeddings for Q, K, and V
        self.linear_q = nn.ModuleList([nn.Linear(d_model, self.d) 
                                       for _ in range(num_heads)])
        self.linear_k = nn.ModuleList([nn.Linear(d_model, self.d)
                                       for _ in range(num_heads)])
        self.linear_v = nn.ModuleList([nn.Linear(d_model, self.d)
                                       for _ in range(num_heads)])

        self.mha_linear = nn.Linear(d_model, d_model)

    def dot_product_attention(self, Q, K, V, dk=4):
        # matmul Q and K
        QK = torch.matmul(Q, K.T)

        matmul_scaled = QK / np.sqrt(dk)
    
        attention_weights = F.softmax(matmul_scaled, dim=-1)

        output = torch.matmul(attention_weights, V)

        return output, attention_weights

    def forward(self, x):
        Q = [linear_q(x) for linear_q in self.linear_q]
        K = [linear_k(x) for linear_k in self.linear_k]
        V = [linear_v(x) for linear_v in self.linear_v]

        outputs_per_head = []
        attn_per_head = []

        for Q_, K_, V_ in zip(Q, K, V):
            output, attn = self.dot_product_attention(Q_, K_, V_)

            outputs_per_head.append(output)
            attn_per_head.append(attn)

        output = torch.cat(outputs_per_head, -1)

