import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, query, key, values):
        pass


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers=12, embeddings_dim=768, num_heads=16, dropout=0.1):
        super().__init__()
        self.encoders = nn.ModuleList([TransformerEncoderLayer(embeddings_dim, num_heads, dropout) for _ in range(num_layers)])
    
    def forward(self, input):
        for encoder_block in self.encoders:
            input = encoder_block(input)
        return input

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embeddings_dim=768, num_heads=16, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embeddings_dim)
        self.msa = nn.MultiheadAttention(embeddings_dim, num_heads=num_heads, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(embeddings_dim), nn.Linear(embeddings_dim, embeddings_dim*4), nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(embeddings_dim*4, embeddings_dim),
                                 nn.Dropout(dropout))

    def forward(self, input):
        input = self.layer_norm(input)
        input = self.msa(input, input, input)[0] + input
        return self.mlp(input) + input