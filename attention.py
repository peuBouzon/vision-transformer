import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embeddings_dim=768, num_heads=16, dropout=0.1):
        super().__init__()
        assert embeddings_dim % num_heads == 0, 'The embeddings dim must be divisible by the number of heads'
        head_input_dim = embeddings_dim // num_heads
        self.num_heads = num_heads
        self.linear_queries = nn.ModuleList([nn.Linear(embeddings_dim, head_input_dim) for _ in range(num_heads)])
        self.linear_keys = nn.ModuleList([nn.Linear(embeddings_dim, head_input_dim) for _ in range(num_heads)])
        self.linear_values = nn.ModuleList([nn.Linear(embeddings_dim, head_input_dim) for _ in range(num_heads)])
        self.scaled_dot_attentions = nn.ModuleList([ScaledDotProductAttention(embeddings_dim) for _ in range(num_heads)])
        self.linear_output = nn.Linear(embeddings_dim, embeddings_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, values):
        outs = []
        for i in range(self.num_heads):
            q = self.linear_queries[i](query)
            k = self.linear_keys[i](key)
            v = self.linear_values[i](values)
            outs.append(self.scaled_dot_attentions[i](q, k, v))
        return self.linear_output(torch.cat(outs, dim=-1))
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self, embeddings_dim=768):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.scale_factor = embeddings_dim ** 0.5
    
    def forward(self, query, key, value):
        return ((query @ key.transpose(-1, -2)) / self.scale_factor) @ value