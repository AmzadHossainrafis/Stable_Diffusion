import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from SDiffusion.components.models import VAE, Discriminator
import tqdm
from SDiffusion.components.LPIPS import LPIPS


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
    train_dataset, batch_size=27, num_workers=4, shuffle=True
)


class VAETrainer:
    """
    VAE_trainer is responsible for training a Variational Autoencoder (VAE) model along with a Discriminator model.
    
    Args:
        config (dict): Configuration dictionary containing training parameters.
        model (nn.Module): The VAE model to be trained.
        disc_model (nn.Module): The Discriminator model to be trained.
    def train(self, train_loader):
   
        Train the VAE model using the provided data loader.

        Parameters:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
     
        disc_optimizer (torch.optim.Optimizer): Optimizer for the Discriminator model.
    """
    def __init__(self, config, model, disc_model, optimizer, disc_optimizer):
        """
        Initialize the VAE_trainer class.

        Args:
            config (dict): Configuration dictionary containing training parameters.
            model (nn.Module): The VAE model to be trained.
            disc_model (nn.Module): The Discriminator model to be trained.
            optimizer (torch.optim.Optimizer): Optimizer for the VAE model.
            disc_optimizer (torch.optim.Optimizer): Optimizer for the Discriminator model.
        """
        self.config = config
        self.model = model.to(self.config["device"])
        self.disc_model = disc_model.to(config["device"])
        self.optimizer = optimizer
        self.disc_optimizer = disc_optimizer

    def train(self, train_loader):
        """
        Train the VAE model using the provided training data loader.

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.

        Returns:
            None

        The training process includes:
        - Computing the mean squared error (MSE) loss between the reconstructed and original images.
        - Computing the Learned Perceptual Image Patch Similarity (LPIPS) loss.
        - Computing the discriminator loss using binary cross-entropy (BCE) loss.
        - Accumulating and backpropagating the losses.
        - Updating the model and discriminator parameters based on the accumulated gradients.
        - Saving the trained model to the specified path in the configuration.

        Note:
            - The discriminator starts training after a specified number of epochs.
            - The losses are accumulated and the optimizer steps are performed after a specified number of steps.
        """
        mse_loss_fn = nn.MSELoss()
        lpips_loss_fn = LPIPS().eval().to(self.config['device'])          
        disc_loss_fn = nn.BCELoss()
        losses, gen_losses, recon_losses, perseptual_loss, disc_losses = [], [], [], [], []

        #model training  loop 
        self.model.train()
        for epoch in range(self.config["epochs"]):
            for i, (images, _) in enumerate(tqdm.tqdm(train_loader)):
                images = images.to(self.config["device"])
                output = self.model(images) 
                recon_loss = mse_loss_fn(output, images) 
                perseptual_loss = torch.mean(lpips_loss_fn(output, images))


                if i >= self.config["disc_start"]:
                    Disc_output = self.disc_model(images) 
                    disc_loss = disc_loss_fn(Disc_output, torch.ones_like(Disc_output))
                    disc_losses.append(disc_loss.item())


                recon_losses.append(recon_loss.item())
                recon_loss = recon_loss / self.config["encoder_acc_steps"]

                g_loss = recon_loss

                if i >= self.config["disc_start"]: 
                    dis_fake = self.disc_model(output)
                    disc_loss = disc_loss_fn(dis_fake,  torch.ones(dis_fake.shape, device=self.config["device"]))
                    gen_losses.append(self.config["disc_loss_weight"] * disc_loss.item())
                    g_loss += self.config["disc_loss_weight"] * disc_loss

                perseptual_loss.append(self.config["perceptual_loss_weight"] * perseptual_loss.item())
                g_loss += self.config["perceptual_loss_weight"] * perseptual_loss / self.config["encoder_acc_steps"]

                losses.append(g_loss.item())
                g_loss.backward()

                # discriminator training 
            
                if i >= self.config["disc_start"]: 
                    fake = output.detach() 
                    fack = self.disc_model(fake)
                    real= self.disc_model(images)

                    fake_loss = disc_loss_fn(fack, torch.zeros(fack.shape, device=self.config["device"]))
                    real_loss = disc_loss_fn(real, torch.ones(real.shape, device=self.config["device"]))
                    discriminator_loss = self.config['nomalizer_weight']*(real_loss + fake_loss) / 2


                    disc_losses.append(discriminator_loss.item())
                    discriminator_loss = discriminator_loss / self.config["encoder_acc_steps"]

                    discriminator_loss.backward()

                if i % self.config["encoder_acc_steps"] == 0:
                    self.disc_optimizer.step()
                    self.disc_optimizer.zero_grad()

                if i % self.config["encoder_acc_steps"] == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            self.model.step()
            self.model.zero_grad()
            self.disc_model.step()
            self.disc_model.zero_grad()

            if len(losses) > 0:
                print(
                    f"Epoch {epoch+1}/{self.config['epochs']}, "
                    f"Loss: {sum(losses)/len(losses):.4f}, "
                    f"Recon Loss: {sum(recon_losses)/len(recon_losses):.4f}, "
                    f"Perseptual Loss: {sum(perseptual_loss)/len(perseptual_loss):.4f}, "
                    f"Disc Loss: {sum(disc_losses)/len(disc_losses):.4f}"
                )

            else:
                print(f"Epoch {epoch+1}/{self.config['epochs']},
                    Loss: {sum(losses)/len(losses):.4f},
                    Recon Loss: {sum(recon_losses)/len(recon_losses):.4f},
                    Perseptual Loss: {sum(perseptual_loss)/len(perseptual_loss):.4f},
                    , Disc Loss: {sum(disc_losses)/len(disc_losses):.4f}"
                )
            
            #save the model 
            if epoch % self.config["save_interval"] == 0:
                torch.save(self.model.state_dict(), self.config["model_path"])
                torch.save(self.disc_model.state_dict(), self.config["disc_model_path"])
                print("Model saved")

            print("Training completed")

                    

            


            

