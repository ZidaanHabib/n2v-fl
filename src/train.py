import time
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from utils.build import load_dataset, setup_loss, setup_optimizer, train_step, test_step, seed, set_device
from models.unet import UNet

import torch
from pathlib import Path

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

    

    Path("checkpoints").mkdir(parents=True, exist_ok=True)


    epochs = cfg.train.epochs
    best_val_loss = float("inf")
    start_time = time.perf_counter()
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n---------")
        train_step(data_loader=train_loader, 
            model=model, 
            loss_fn=loss_fn,
            optimizer=optim,
            device=device
  
        )
        avg_loss, avg_psnr = test_step(data_loader=test_loader,
            model=model,
            loss_fn=loss_fn,
            device=device
        )

        torch.save(
            {
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optim.state_dict(),
                "avg_loss":    avg_loss,
                "avg_psnr":    avg_psnr
            },
            "checkpoints/last.pth",
        )

        # If it’s the best so far, also save “best.pth”
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            torch.save(
                model.state_dict(),  # you can omit optimizer if only inference
                "checkpoints/best.pth",
            )
    total_time = time.perf_counter() - start_time
    print(f"Training time: {(total_time / 60):.2f} minutes")

    

if __name__ == "__main__":
    main()