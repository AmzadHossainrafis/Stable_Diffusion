
import torchvision 
import torch.nn as nn 
from torchinfo import summary

class clip_imagent_encoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = torchvision.models.resnet50(pretrained=True)   
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.model.eval()
        self.model.requires_grad_(False)
        self.reshape = nn.Flatten()
        # self.model last layer shape is (batch, 2048, 7, 7)
        self.linner = nn.Linear(2048*7*7, 512)

    def forward(self, x):
        x = self.model(x)
        x = self.reshape(x)
        x = self.linner(x)
    
        return x
        




# if __name__ == '__main__':
#     import torch
#     model = clip_imagent_encoder()
#     print(f'model output shape: {model(torch.randn(1, 3, 224, 224)).shape}')
#     print(summary(model, input_size=(1, 3, 224, 224), verbose=1))