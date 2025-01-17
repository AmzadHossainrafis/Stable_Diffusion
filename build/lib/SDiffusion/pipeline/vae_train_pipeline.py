import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torchvision
from SDiffusion.components.models import VAE 
import tqdm  
from SDiffusion.components.LPIPS import LPIPS 
#kl divergence




Dataset_dir: str = r"/home/amzad/Downloads/celb_face/img_align_celeba/"

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((64, 64)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# load the dataset
train_dataset = torchvision.datasets.ImageFolder(root=Dataset_dir, transform=transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=80, num_workers=4, shuffle=True
)



class VAE_trainer:
    def __init__(self, config, model, optimizer):
        self.model = model.to(config['device'])
        self.optimizer = optimizer
        self.config = config

    def train(self, train_loader):
        self.model.train()
        for epoch in range(self.config['epochs']):
            for i, (x, _) in enumerate(tqdm.tqdm(train_loader)):
                x = x.to(self.config['device'])
                noise = torch.randn(x.size(0), 256, 8, 8).to(self.config['device'])
                x_hat = self.model(x, noise)

                mse_loss = nn.MSELoss(x_hat, x)
                kl_loss = nn.KLDivLoss(x_hat, x)
                LPIPS_loss = LPIPS(x_hat, x)
                loss = mse_loss + kl_loss + LPIPS_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print(f'epoch: {epoch}, batch: {i}, loss: {loss.item()}')
        torch.save(self.model.state_dict(), self.config['model_save_path'])
        print('model saved successfully')



if __name__ == '__main__':
    model = VAE().to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
   
    config = {
        'epochs': 20,
        'device': 'cuda',
        'model_save_path': '/home/amzad/Desktop/stable_diffusion/artifacts/vae_model.pth'
    }
    trainer = VAE_trainer(config, model, optimizer)
    trainer.train(train_loader)