import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from SDiffusion.components.models import VAE, Discriminator
import tqdm
from SDiffusion.components.LPIPS import LPIPS
import matplotlib.pyplot as plt


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
        lpips_loss_fn = LPIPS().to(self.config["device"])
        disc_loss_fn = nn.MSELoss()
        losses, gen_losses, recon_losses, perseptual_losses, disc_losses = (
            [],
            [],
            [],
            [],
            [],
        )

        # model training  loop
        self.model.train()
        for epoch in range(self.config["epochs"]):
            for i, (images, _) in enumerate(tqdm.tqdm(train_loader)):
                images = images.to(self.config["device"])
                noise = torch.randn(images.shape[0], 256,8,8).to(self.config["device"])
                output = self.model(images, noise)
                recon_loss = mse_loss_fn(output, images)
                perseptual_loss = torch.mean(lpips_loss_fn(output, images))

                if i % self.config["inface"] == 0:
                  # make a simple prediction 
                   torchvision.utils.make_grid(output.permute(0, 2, 3, 1).detach().cpu(), nrow=5) 
                   torchvision.utils.save_image(output, f"fig/output_{i}.png" ,nrow= 5,normalize=True)
                 

                if i >= self.config["disc_start"]:
                    Disc_output = self.disc_model(images)
                    disc_loss = disc_loss_fn(Disc_output, torch.ones(Disc_output.shape, device=self.config["device"]))
                    disc_losses.append(disc_loss.item())

                recon_losses.append(recon_loss.item())
                recon_loss = recon_loss / self.config["encoder_acc_steps"]

                g_loss = recon_loss

                if i >= self.config["disc_start"]:
                    dis_fake = self.disc_model(output)
                    disc_loss = disc_loss_fn(
                        dis_fake,
                        torch.ones(dis_fake.shape, device=self.config["device"]),
                    )
                    gen_losses.append(
                        self.config["disc_loss_weight"] * disc_loss.item()
                    )
                    g_loss += self.config["disc_loss_weight"] * disc_loss

                perseptual_losses.append(
                    self.config["perceptual_loss_weight"] * perseptual_loss.item()
                )
                g_loss += (
                    self.config["perceptual_loss_weight"]
                    * perseptual_loss
                    / self.config["encoder_acc_steps"]
                )

                losses.append(g_loss.item())
                g_loss.backward()

                # discriminator training

                if i >= self.config["disc_start"]:
                    fake = output.detach()
                    fack = self.disc_model(fake)
                    real = self.disc_model(images)

                    fake_loss = disc_loss_fn(
                        fack, torch.zeros(fack.shape, device=self.config["device"])
                    )
                    real_loss = disc_loss_fn(
                        real, torch.ones(real.shape, device=self.config["device"])
                    )
                    discriminator_loss = (
                        self.config["nomalizer_weight"] * (real_loss + fake_loss) / 2
                    )

                    disc_losses.append(discriminator_loss.item())
                    discriminator_loss = (
                        discriminator_loss / self.config["encoder_acc_steps"]
                    )

                    discriminator_loss.backward()

                if i % self.config["encoder_acc_steps"] == 0:
                    self.disc_optimizer.step()
                    self.disc_optimizer.zero_grad()

                if i % self.config["encoder_acc_steps"] == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # self.model.step()
            # self.model.zero_grad()
            # self.disc_model.step()
            # self.disc_model.zero_grad()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.disc_optimizer.step()
            self.disc_optimizer.zero_grad()

            

            if len(losses) > 0:
                print(
                    "Epoch {}/{}, recon_loss: {:.4f}, Perseptual Loss: {}, Disc Loss: {:.4f}".format(
                        epoch + 1,
                        self.config["epochs"],
                        sum(recon_losses) / len(recon_losses),
                        sum(perseptual_losses) / len(perseptual_losses),
                        sum(disc_losses) / len(disc_losses),
                    )
                )

            else:
                print(
                    f"Epoch {epoch+1}/{self.config['epochs']} , recon_loss: {recon_loss.item():.4f}"
                    f"Perseptual Loss: {perseptual_loss.item():.4f} , Disc Loss: {disc_loss.item():.4f}"
                )

            # save the model
            if epoch % self.config["save_interval"] == 0:
                torch.save(self.model.state_dict(), self.config["model_path"])
                torch.save(self.disc_model.state_dict(), self.config["disc_model_path"])
                print("Model saved")

            print("Training completed")


if __name__ == "__main__":

    


    config = {
        "batch_size": 35,
        "dataset_dir": r"/home/amzad/Downloads/celb_face/img_align_celeba/",
        "resize": (64, 64),
        "normalize": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        "workers": 4,
        "shuffle": True,
        "lr": 1e-3,
        "inface": 10,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "epochs": 500,
        "encoder_acc_steps": 1,
        "disc_start": 200,
        "disc_loss_weight": 0.1,
        "perceptual_loss_weight": 0.1,
        "nomalizer_weight": 0.1,
        "save_interval": 10,
        "model_ckpt_dir": "model_checkpoint",
        "model_path": "artifats/vae_model.pth",
        "disc_model_path": "artifacts/disc_model.pth",

    }
    

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((config["resize"])),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((config["normalize"][0]), (config["normalize"][1])),
        ]
    )

    # load the dataset
    train_dataset = torchvision.datasets.ImageFolder(root=config["dataset_dir"], transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], num_workers=config['workers'], shuffle=config['shuffle']
    )


    model = VAE()
    disc_model = Discriminator()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    disc_optimizer = torch.optim.Adam(disc_model.parameters(), lr=config["lr"])
    trainer = VAETrainer(config, model, disc_model, optimizer, disc_optimizer)
    trainer.train(train_loader)
