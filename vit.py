import torch
from torch import nn
from transformer import TransformerEncoder
import einops
from torchinfo import summary

class PatchEmbedding(nn.Module):
        # "To handle 2D images, we reshape the image x ∈ R^(H×W×C) into a
        # sequence of flattened 2D patches xp ∈ R^(N ×(P²·C)) , where (H, W) is the resolution of the original
        # image, C is the number of channels, (P, P) is the resolution of each image patch, and N = HW/P²
        # is the resulting number of patches, which also serves as the effective input sequence length for the
        # Transformer".
    def __init__(self, embedding_dim=768, n_channels=3, img_size=224, patch_size=16, dropout=0.1):
        super().__init__()
        N = int(img_size * img_size / patch_size ** 2)
        self.positional_embeddings = nn.Parameter(torch.randn((1, N+1, embedding_dim)))
        self.class_token = nn.Parameter(torch.randn((1, 1, embedding_dim)))
        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=embedding_dim, 
                              stride=patch_size, kernel_size=patch_size, padding=0) # clever way to apply a linear projection to patches of the input image  
        self.dropout = nn.Dropout(dropout)
        self.embedding_dim = embedding_dim

    def forward(self, image:torch.Tensor):
        # the class embeddings are the same for the entire batch
        class_embeddings = einops.repeat(self.class_token, '() n d -> batch_size n d', batch_size=image.shape[0])
        patch_embeddings = einops.rearrange(self.conv(image), 'b c h w -> b (h w) c')
        return self.dropout(torch.cat([class_embeddings, patch_embeddings], dim=1) + self.positional_embeddings)

class ViT(nn.Module):
    def __init__(self, n_classes, num_layers=12, num_heads=16, embedding_dim=768, img_size=224, n_channels=3, dropout=0.1, patch_size=16):
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            patch_size=patch_size,
            img_size=img_size,
            embedding_dim=embedding_dim,
            n_channels=n_channels,
            dropout=dropout
        )
        self.pre_transformer_dropout = nn.Dropout(dropout)
        self.transformer = TransformerEncoder(num_layers, embedding_dim, num_heads, dropout)
        
        # "The classification head is implemented by a MLP with one hidden layer 
        # at pre-training time and by a single linear layer at fine-tuning time."
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, n_classes)
        )

    def forward(self, image):
        transfomer_out = self.transformer(self.patch_embedding(image))
        return self.classifier(transfomer_out[:, 0]) # only the [class] state is used as input

model = ViT(n_classes=10, )


if __name__ == '__main__':
    print(summary(model, input_size=(1,3,224,224)))
    data = torch.randn((1, 3, 224, 224))
    out = model(data)
    print(out.shape)