import torch.nn as nn
from torchinfo import summary
import torch 


class CLIPEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.text_embedding = nn.Embedding(self.config["vocab"], self.config["n_emb"])
        self.positional_embedding = nn.Parameter(torch.zeros(self.config["token"], self.config["n_emb"]))


    def forward(self, x): 
        x = self.text_embedding(x)
        x = x + self.positional_embedding
        return x
    
class CLIPLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer_norm1 = nn.LayerNorm(self.config["n_emb"])
        self.self_attn = nn.MultiheadAttention(self.config["n_emb"], self.config["n_head"])
        self.layer_norm2 = nn.LayerNorm(self.config["n_emb"])
        self.linear1 = nn.Linear(self.config["n_emb"], 4 * self.config["n_emb"])
        self.linear2 = nn.Linear(4 * self.config["n_emb"], self.config["n_emb"])

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
    def __init__(self, config,):
        super().__init__()
        self.config = config
        self.embedding = CLIPEmbedding(self.config)
        # self.transformer_encoder = nn.TransformerEncoderLayer(self.config["n_emb"], self.config["n_head"])
        # self.transformer = nn.Transformer(self.config["n_emb"], self.config["n_head"], self.config["n_layer"])
        self.layers = nn.ModuleList([CLIPLayer(self.config) for _ in range(self.config["n_layer"])])
        self.layer_norm = nn.LayerNorm(self.config["n_emb"])

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        return x
        


    

if __name__ == '__main__':
    config = {
        "vocab": 48000,
        "n_emb": 512,
        "n_head": 8,
        "token": 100,
        "n_layer": 12,
    }
    model = CLIP(config).to('cuda')
    x = torch.randint(0, 10000, (1, 100)).long().to('cuda')



    summary(model, input_data=x)
    print(model(x).shape)
    # print(model(x).shape)