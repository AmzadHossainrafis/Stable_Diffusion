import torch.nn as nn
from torchinfo import summary
import torch 
from transformers import BertTokenizer


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
        


    


import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel, BertTokenizer

class ImageEncoder(nn.Module):
    def __init__(self, embed_size):
        super(ImageEncoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]  # Remove the last fully connected layer
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    
class TextEncoder(nn.Module):
    def __init__(self, embed_size):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.embed = nn.Linear(self.bert.config.hidden_size, embed_size)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        features = self.embed(pooled_output)
        return features

class CLIPModel(nn.Module):
    def __init__(self, embed_size):
        super(CLIPModel, self).__init__()
        self.image_encoder = ImageEncoder(embed_size)
        self.text_encoder = TextEncoder(embed_size)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)  # Initialized as in the original CLIP
        
    def forward(self, images, input_ids, attention_mask):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(input_ids, attention_mask)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text
    



    
# if __name__ == '__main__':
#     config = {
#         "vocab": 48000,
#         "n_emb": 224,
#         "n_head": 8,
#         "token": 100,
#         "n_layer": 12,
#     }

# #     def pos_encoding(t, channels):
# #         inv_freq = 1.0 / (
# #             10000
# #             ** (torch.arange(0, channels, 2, device='cuda').float() / channels)
# #         )
# #         pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
# #         pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
# #         pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
# #         return pos_enc
    
#     model = CLIP(config).to('cuda')
#     #random dami text batch shape (batch, 
    
#     embedding = CLIPEmbedding(config).to('cuda')
#     text = "hello world this is a test text for the model to learn the embedding <UNK>" 
#     # PADD IT TO MAX TOKENS 100

#     for i in range(100 - len(text.split())):
#         text += " <UNK>"
#     tokens = text.split()
#     # create embedding 
#     tokens_id_list = {
#         token: i for i, token in enumerate(tokens)
#     }

#     tokens_id = [tokens_id_list[token] for token in tokens]
#     tokens_id = torch.tensor(tokens_id).unsqueeze(0).to('cuda')
#     demo = embedding(tokens_id) 
#     print(demo.shape)