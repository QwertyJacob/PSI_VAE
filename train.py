import hydra
from omegaconf import DictConfig, OmegaConf
import torch.optim as optim
from PSI.utils import train_vae, get_manual_dataloaders, init_wandb
from PSI.models import VAE
import torch


@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig):

    if cfg.override != "":
        try:
            # Load the variant specified from the command line
            config_overrides = OmegaConf.load(hydra.utils.get_original_cwd() + f'/conf/overrides/{cfg.override}.yaml')
            # Merge configurations, with the variant overriding the base config
            cfg = OmegaConf.merge(cfg, config_overrides)
        except:
            print('Unsuccesfully tried to use the configuration override: ',cfg.override)


    latent_dim = cfg.latent_dim
    batch_size = cfg.batch_size
    epochs = cfg.epochs
    learning_rate = cfg.learning_rate
    device = torch.device(cfg.device)
    wb = cfg.wandb
    
    train_dataloader, test_dataloader = get_manual_dataloaders(batch_size)
    vae = VAE(latent_dim)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    init_wandb(cfg)
    train_vae(vae, train_dataloader, test_dataloader, optimizer, epochs, device, wb)

if __name__ == "__main__":
    main()