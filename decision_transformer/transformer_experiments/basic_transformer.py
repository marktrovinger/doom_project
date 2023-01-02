import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def dot_product_attention(Q, K, V, dk=4):
    # matmul Q and K
    QK = torch.matmul(Q, K.T)

    matmul_scaled = QK / np.sqrt(dk)
    
    attention_weights = F.softmax(matmul_scaled, dim=-1)

    output = torch.matmul(attention_weights, V)

    return output, attention_weights


def print_attention(Q, K, V, n_digits=3):
    temp_out, temp_attn = dot_product_attention(Q, K, V)
    temp_out, temp_attn = temp_out.numpy(), temp_attn.numpy()

    print(f"Attention weights are: {np.round(temp_out, n_digits)}")
    print(f"Output weights are: {np.round(temp_attn, n_digits)}")


temp_k = torch.Tensor([[10, 0, 0],
                        [0, 10, 0],
                        [0, 0, 10],
                        [0, 0, 10]])

temp_v = torch.Tensor([[1, 0, 1], 
                       [10, 0, 2],
                       [100, 5, 0],
                       [1000, 6, 0]])

temp_q = torch.Tensor([[0, 10, 0]])
print_attention(temp_q, temp_k, temp_v)
    


