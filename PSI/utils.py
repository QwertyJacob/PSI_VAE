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
from torchmetrics.image.fid import FrechetInceptionDistance


def init_wandb(cfg):
    wandb.init(
        project="VAE", 
        mode=("online" if cfg.wandb else "disabled"), 
        config=dict(cfg))


def get_manual_dataloaders(batch_size):
    transform = transforms.Compose([
        transforms.CenterCrop(178),  # Crop face region
        transforms.Resize(64),  # Resize to 64x64
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(root="./data/celeba", transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader


# Loss function
def loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld


# Training loop
def train_vae(model, train_dataloader, test_dataloader, optimizer, epochs, device, wb):
    model.to(device)
    step = 0
    fid = FrechetInceptionDistance(feature=2048).to(device)  # Using 2048 for better accuracy

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        train_loss = 0
        for x, _ in tqdm(train_dataloader):
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
                visualize_reconstructions(model, train_dataloader, device, step, wb)
                
        print(f"Epoch {epoch + 1}, Loss: {train_loss / len(train_dataloader.dataset):.4f}")
        
        compute_fid(fid, model, test_dataloader, device, step)


def compute_fid(fid, model, test_dataloader, device, step):
    # FID computation
    model.eval()
    with torch.no_grad():
        fid.reset()  # Clear previous calculations
        real_images = []
        fake_images = []

        print('Validation...')
        
        # Collect real and fake images
        for real_batch, _ in tqdm(test_dataloader):
            real_batch = real_batch.to(device)
            real_images.append(real_batch)
            
            fake_batch, _, _ = model(real_batch)  # Generate fake images
            fake_images.append(fake_batch)

            # Stop collecting after enough samples
            if len(real_images) > 10:  # Adjust this number for more accuracy
                break

        # Add images to FID metric
        for real, fake in zip(real_images, fake_images):
            fid.update(real, real=True)
            fid.update(fake, real=False)

        # Compute FID score
        fid_score = fid.compute().item()
        print(f"FID Score after Epoch {epoch + 1}: {fid_score:.4f}")
        wandb.log({"FID": fid_score}, step=step)



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


