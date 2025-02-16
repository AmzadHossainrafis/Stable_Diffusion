import torch
import torch.nn as nn
from SDiffusion.components.clip import CLIPModel
from SDiffusion.components.data_loader import ImageTextDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BertModel, BertTokenizer
import tqdm
import pandas as pd

# warnings.filterwarnings("ignore")
import warnings

warnings.filterwarnings("ignore")
import os


def contrastive_loss(logits_per_image, logits_per_text):
    labels = torch.arange(logits_per_image.size(0), device=logits_per_image.device)
    loss_img = nn.CrossEntropyLoss()(logits_per_image, labels)
    loss_txt = nn.CrossEntropyLoss()(logits_per_text, labels)
    return (loss_img + loss_txt) / 2


def train_clip(model, dataloader, optimizer, device, epochs=50):
    model.train()
    for epoch in range(epochs):
        losses = []
        pbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (images, input_ids, attention_mask) in pbar:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            optimizer.zero_grad()
            logits_per_image, logits_per_text = model(images, input_ids, attention_mask)
            loss = contrastive_loss(logits_per_image, logits_per_text)
            losses.append(loss.item())
            loss.backward()
            # ddisplay(loss) in tqdm progress bar
            pbar.set_description(
                f"Epoch {epoch+1}, Loss: {torch.tensor(losses).mean():.4f}"
            )
            optimizer.step()

        print(f"Epoch {epoch+1}-{i}, Loss: {loss.item()}")


# Define transforms
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


data = pd.read_csv("/home/amzad/Desktop/stable_diffusion/dataset/flicker.csv")

image_paths = data["image_id"].tolist()
texts = data["text"].tolist()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Dataset and DataLoader
dataset = ImageTextDataset(image_paths, texts, tokenizer, transform=transform)
dataloader = DataLoader(dataset, batch_size=25, shuffle=True)

# Model, Optimizer, and Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel(embed_size=512).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Train the model
train_clip(model, dataloader, optimizer, device, epochs=5)


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BertTokenizer
import tqdm
import warnings

from SDiffusion.components.clip import CLIPModel
from SDiffusion.components.data_loader import ImageTextDataset

# Suppress warnings
warnings.filterwarnings("ignore")


class CLIPTrainer:
    def __init__(self, config: dict, transform=None):
        self.config = config
        self.device = config["device"]
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.transform = transform
        self.model = CLIPModel(config["embed_size"]).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config["learning_rate"]
        )
        self.dataloader = self._create_dataloader(config["data_path"])

    def _create_dataloader(self, data_path: list, text_path: list):
        dataset = ImageTextDataset(data_path, text_path, self.tokenizer, self.transform)
        return DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True)

    def contrastive_loss(self, logits_per_image, logits_per_text):
        labels = torch.arange(logits_per_image.size(0), device=logits_per_image.device)
        loss_img = nn.CrossEntropyLoss()(logits_per_image, labels)
        loss_txt = nn.CrossEntropyLoss()(logits_per_text, labels)
        return (loss_img + loss_txt) / 2

    def train(self):
        for epoch in range(self.config["num_epochs"]):
            self.model.train()
            losses = []
            pbar = tqdm.tqdm(enumerate(self.dataloader), total=len(self.dataloader))
            for i, batch in pbar:
                images, input_ids, attention_mask = batch
                images = images.to(self.device)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                self.optimizer.zero_grad()
                logits_per_image, logits_per_text = self.model(
                    images, input_ids, attention_mask
                )
                loss = self.contrastive_loss(logits_per_image, logits_per_text)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                total_loss += loss.item()
                pbar.set_description(
                    f"Epoch {epoch+1}, Loss: {torch.tensor(losses).mean():.4f}"
                )

            avg_loss = total_loss / len(self.dataloader)
            print(
                f"Epoch [{epoch+1}/{self.config['num_epochs']}], Loss: {avg_loss:.4f}"
            )

    def save_model(self, epoch: int, interval: int):

        if epoch % interval == 0:
            # Save the model
            torch.save(
                self.model.state_dict(),
                os.path.join(self.config["artifact"], "clip_model.pth"),
            )
            print("Model saved successfully.")
            # text encoder
            torch.save(
                self.model.text_encoder.state_dict(),
                os.path.join(self.config["artifact"], "clip_text_encoder.pth"),
            )
            print("Text encoder saved successfully.")
            # image encoder
            torch.save(
                self.model.image_encoder.state_dict(),
                os.path.join(self.config["artifact"], "clip_image_encoder.pth"),
            )
            print("Image encoder saved successfully.")

        # Save the model


if __name__ == "__main__":
    config = {
        "batch_size": 32,
        "num_epochs": 10,
        "learning_rate": 1e-4,
        "embed_size": 512,
        "interval": 5,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_path": "/home/amzad/Desktop/stable_diffusion/dataset/flicker.csv",
        "artifact": "/home/amzad/Desktop/stable_diffusion/artifacts",
        "size": (224, 224),
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }

    #     trainer = CLIPTrainer(config)
    #     trainer.train()

    #     trainer.save_model(10,5)
    data = pd.read_csv("/home/amzad/Desktop/stable_diffusion/dataset/flicker.csv")
    image_paths = data["image_id"].tolist()
    texts = data["text"].tolist()
    transform = transforms.Compose(
        [
            transforms.Resize(config["size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=config["mean"], std=config["std"]),
        ]
    )

    trainer = CLIPTrainer(config, transform)
    trainer.train()
