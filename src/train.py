import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from utils.build import load_dataset, setup_loss, setup_optimizer, train_step, test_step, seed, set_device
from models.unet import UNet



@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg: DictConfig):

    device = set_device()

    # do the global seeding for RNGs
    seed(42)

    #load dataset:
    train_loader, test_loader = load_dataset(batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, patch_size=cfg.data.patch_size)
    
    #instantiate model
    model = UNet(1,cfg.model.base_channels, cfg.model.depth)

    #set up loss and optimizer
    loss_fn = setup_loss()
    optim = setup_optimizer(model_params=model.parameters(), type=cfg.optimizer.type, lr=cfg.optimizer.lr, betas=cfg.optimizer.betas)

    epochs = cfg.train.epochs
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n---------")
        train_step(data_loader=train_loader, 
            model=model, 
            loss_fn=loss_fn,
            optimizer=optim,
  
        )
        test_step(data_loader=test_loader,
            model=model,
            loss_fn=loss_fn,
        )


    

if __name__ == "__main__":
    main()