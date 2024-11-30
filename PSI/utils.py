import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import deeplake
from torchvision.datasets import ImageFolder
import wandb
from tqdm import trange, tqdm

def init_wandb(cfg):
    wandb.init(
        project="VAE", 
        mode=("online" if cfg.wandb else "disabled"), 
        config=dict(cfg))


def get_manual_dataloader(batch_size):
    transform = transforms.Compose([
        transforms.CenterCrop(178),  # Crop face region
        transforms.Resize(64),  # Resize to 64x64
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(root="./data/celeba", transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Load and preprocess CelebA dataset
def get_dataloader(batch_size):
    transform = transforms.Compose([
        transforms.CenterCrop(178),  # Crop out the face region
        transforms.Resize(64),  # Resize to 64x64
        transforms.ToTensor(),
    ])
    
    dataset = datasets.CelebA(root="./data/celeba", download=False, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


# Loss function
def loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld


# Training loop
def train_vae(model, dataloader, optimizer, epochs, device, wb):
    model.to(device)
    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        train_loss = 0
        for x, _ in tqdm(dataloader):
            step += 1
            x = x.to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = loss_function(recon_x, x, mu, logvar)
            loss.backward()
            wandb.log({"loss": loss}, step=step)
            train_loss += loss.item()
            optimizer.step()

            if step % 500 == 0:
                visualize_reconstructions(model, dataloader, device, step, wb)

        print(f"Epoch {epoch + 1}, Loss: {train_loss / len(dataloader.dataset):.4f}")
        visualize_reconstructions(model, dataloader, device, step, wb)


# Visualize reconstructions
def visualize_reconstructions(model, dataloader, device, step, wb):
    model.eval()
    x, _ = next(iter(dataloader))
    x = x.to(device)
    with torch.no_grad():
        recon_x, _, _ = model(x)
    x = x.cpu()
    recon_x = recon_x.cpu()
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(8):
        axes[0, i].imshow(x[i].permute(1, 2, 0))
        axes[0, i].axis("off")
        axes[1, i].imshow(recon_x[i].permute(1, 2, 0))
        axes[1, i].axis("off")

    plt.tight_layout()

    if wb: wandb.log({f"Reconstructions": wandb.Image(plt)}, step=step, commit=False)
    else: plt.show()

    plt.cla()
    plt.close()


