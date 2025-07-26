import time
from datetime import datetime
import hydra
from omegaconf import DictConfig
import os
from utils.build import  load_distributed_dataset, setup_loss, setup_optimizer, seed, set_device
from models.unet import UNet

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path

@hydra.main(version_base=None, config_path="config", config_name="benchmark")
def main(cfg: DictConfig):

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank       = int(os.environ["RANK"])

    device = set_device()
    
    # associate cuda device to particular gpu for this process
    torch.cuda.set_device(local_rank)

    #Initialise process group
    dist.init_process_group(backend="nccl", init_method="env://") # nccl is the cuda backend for multi-gpu comms, env refers to environment variables which torchrun will populate

    # do the global seeding for RNGs
    seed(42)

    # parameter options to benchmark and compare (from config):
    batch_sizes = cfg.data.batch_sizes #these are per GPU
    num_workers_list = cfg.data.num_workers 

    root = Path().resolve().parent
    data_dir = Path(cfg.data.data_dir) if cfg.data.data_dir else root / "data" / "preprocessed"
    
    #instantiate model
    model = UNet(1,cfg.model.base_channels, cfg.model.depth, batch_norm=cfg.model.batch_norm)
    model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)


    #set up loss and optimizer
    loss_fn = setup_loss()
    optim = setup_optimizer(model_params=model.parameters(), type=cfg.optimizer.type, lr=cfg.optimizer.lr, betas=cfg.optimizer.betas)

    cluster_run_file_name = f"benchmark-run-{datetime.now().strftime("%d_%m_%Hh_%M")}"
    if rank == 0:
        Path("benchmarks").mkdir(parents=True, exist_ok=True)
        with open(f"benchmarks/{cluster_run_file_name}.csv", "w") as f:
            f.write("Batch_size,Num_workers,Epoch_loss,Total_training_time_minutes\n")
    dist.barrier()

    for batch_size in batch_sizes: 
        for num_workers in num_workers_list:
            train_loader, test_loader = load_distributed_dataset(world_size, rank, data_dir, batch_size, num_workers, cfg.data.patch_size, cfg.data.patches_per_image)

            losses = []
            running_loss = 0
            if rank == 0:
                print(f"Batch size: {batch_size},  Num Workers: {num_workers}")
            dist.barrier()
            start_time = time.perf_counter()
            for i in range(2):
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
                    print(f"Rank {rank} Batch {batch} done | Batch loss: {current_loss:.5f} | Batch time: {batch_time/60:.4f}")
            
            dist.barrier()
            total_time = time.perf_counter() - start_time

            epoch_loss   = running_loss   / (2*len(train_loader))
            
            if rank == 0:
                print(f"Train time: {(total_time/60):.5f} ")
                with open(f"benchmarks/{cluster_run_file_name}.csv", "a") as f:
                    f.write(f"{batch_size},{num_workers}, {epoch_loss}, {(total_time/60):.4f}\n")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()