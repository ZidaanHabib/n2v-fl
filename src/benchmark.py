import time
import hydra
from omegaconf import DictConfig


from utils.build import distribute_model, load_dataset, setup_loss, setup_optimizer, train_step, test_step, seed, set_device
from models.unet import UNet

import torch
from pathlib import Path

@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg: DictConfig):

    device = set_device()

    # do the global seeding for RNGs
    seed(42)
    batch_sizes = [64, 128, 256]
    patch_sizes = [256, 512]
    num_workers_list = [8, 16, 32, 48]


    root = Path().resolve().parent
    data_dir = Path(cfg.data.data_dir) if cfg.data.data_dir else root / "data" / "preprocessed"
    
    #instantiate model
    model = UNet(1,cfg.model.base_channels, cfg.model.depth, batch_norm=cfg.model.batch_norm)
    model = distribute_model(model, device)

    #set up loss and optimizer
    loss_fn = setup_loss()
    optim = setup_optimizer(model_params=model.parameters(), type=cfg.optimizer.type, lr=cfg.optimizer.lr, betas=cfg.optimizer.betas)

    Path("benchmarks").mkdir(parents=True, exist_ok=True)

    with open("benchmarks/benchmark.txt", "w") as f:
        f.write("Batch_size,Patch_size,Num_workers,Epoch_loss,Total_training_time_minutes\n")
    for batch_size in batch_sizes: 
        for patch_size in patch_sizes:
            for num_workers in num_workers_list:
                train_loader, test_loader = load_dataset(data_dir, batch_size, num_workers, patch_size, 64)
                losses = []
                running_loss = 0
                print(f"Batch size: {batch_size}, Patch size: {patch_size}, Num Workers: {num_workers}")
                start_time = time.perf_counter()
                for batch, (X, y) in enumerate(train_loader):
                    batch_start_time = time.perf_counter() 
                    X, y = X.to(device), y.to(device)

                    optim.zero_grad()
                    denoised = model(X)
                    loss   = loss_fn(denoised, y)

                    loss.backward()
                    optim.step()

                    current_loss = loss.item()
                    losses.append(current_loss)
                    running_loss   += current_loss

                    batch_time = time.perf_counter() - batch_start_time
                    print(f"Batch {batch} done | Batch loss: {current_loss:.5f} | Batch time: {batch_time/60:.2f}")

                total_time = time.perf_counter() - start_time

                epoch_loss   = running_loss   / len(train_loader)
                print(f"Train loss: {epoch_loss:.5f} ")
                with open("benchmarks/benchmarks.txt", "a") as f:
                    f.write(f"{batch_size},{patch_size}, {num_workers}, {epoch_loss}, {(total_time/60):.2f}\n")

if __name__ == "__main__":
    main()