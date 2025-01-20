import torchvision
import torch.nn as nn
from torchinfo import summary
import torch 


class CLIPEmbedding(nn.Module):
    def __init__(self, vocab, n_emb, token):
        super().__init__()
        self.text_embedding = nn.Embedding(vocab, n_emb)
        self.positional_embedding = nn.Parameter(torch.zeros(token, n_emb))


    def forward(self, x): 
        x = self.text_embedding(x)
        x = x + self.positional_embedding
        return x
    
class CLIPLayer(nn.Module):
    def __init__(self, n_emb, n_head, ):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(n_emb)
        self.self_attn = nn.MultiheadAttention(n_emb, n_head)
        self.layer_norm2 = nn.LayerNorm(n_emb)

        self.linear1 = nn.Linear(n_emb, 4 * n_emb)
        self.linear2 = nn.Linear(4 * n_emb, n_emb)

    def forward(self, x):
        residule = x 

        x = self.layer_norm1(x)
        # q, k ,v = x.chunk(3, dim=-1)
        x, _ = self.self_attn(x, x, x)
        x = x + residule


        residule = x

        x = self.layer_norm2(x)
        x = self.linear1(x)
        x = torch.sigmoid(x * 1.702)
        x = self.linear2(x)
        x = x + residule


        return x
    


class CLIP(nn.Module):
    def __init__(self, vocab, n_emb, n_head, token):
        super().__init__()
        self.embedding = CLIPEmbedding(vocab, n_emb, token)
        self.layers = nn.ModuleList([CLIPLayer(n_emb, n_head) for _ in range(12)])
        self.layer_norm = nn.LayerNorm(n_emb)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        return x
        


    

# if __name__ == '__main__':
#     model = CLIP(100, 512, 8, 100)
#     x = torch.randint(0, 100, (32, 100)).long()
#     summary(model, input_data=x)
#     print(model(x).shape)
#     # print(model(x).shape)