import time
from datetime import datetime
import hydra
from omegaconf import DictConfig
import os
from utils.build import  compute_psnr, setup_loss, setup_optimizer, seed, set_device
from utils.n2v import load_distributed_dataset, mask_batch
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
    patch_sizes = cfg.data.patch_sizes
    patches_per_image = cfg.patches_per_image

    num_workers = cfg.data.num_workers

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
        Path("benchmarks/n2v").mkdir(parents=True, exist_ok=True)
        with open(f"benchmarks/n2v/{cluster_run_file_name}.csv", "w") as f:
            f.write("Batch_size,Patch size,Patches_per_image,Epoch_loss,Total_training_time_minutes\n")
    dist.barrier()

    for batch_size in batch_sizes: 
        for (patch_size, num_patches) in zip(patch_sizes, patches_per_image):
            train_loader, test_loader = load_distributed_dataset(world_size, rank, data_dir, batch_size, 16, patch_size, num_patches, has_ground_truth=False)

            running_loss = torch.tensor(0, dtype=torch.float32, device=device, requires_grad=False)
            losses = torch.zeros(len(train_loader),device=device,requires_grad=False) 
            running_psnr = torch.tensor(0, dtype=torch.float32, device=device, requires_grad=False)
            psnr_list = torch.zeros(len(train_loader), device=device, requires_grad=False)

            if rank == 0:
                print(f"Batch size: {batch_size},  Num Workers: {num_workers}")
            dist.barrier()
            start_time = time.perf_counter()
            for i in range(2):
                for batch, (patches, clean) in enumerate(train_loader):
                    batch_start_time = time.perf_counter() 
                    # perform masking
                    X, y, mask = mask_batch(patches, 64)
                    
                    X, y, mask = X.to(device), y.to(device), mask.to(device)

                    optim.zero_grad()
                    denoised = model(X)
                    loss   = loss_fn(denoised[mask], y[mask])

                    loss.backward()
                    optim.step()

                    current_loss = loss.item()
                    losses[batch] = current_loss
                    running_loss   += current_loss

                    with torch.inference_mode():
                        current_psnr = compute_psnr(denoised, y, clean, 1.0).item()
                    psnr_list[batch] = current_psnr
                    running_psnr += current_psnr

                    batch_time = time.perf_counter() - batch_start_time
                    print(f"Rank {rank} Batch {batch} done | Batch loss: {current_loss:.5f} | Batch PSNR: {current_psnr} | Batch time: {batch_time/60:.4f}")
            
                # Purely added to simulate real training operations:
                dist.all_reduce(losses, op=dist.ReduceOp.AVG)
                dist.all_reduce(psnr_list, op=dist.ReduceOp.AVG)

            total_time = time.perf_counter() - start_time

            epoch_loss   = running_loss   / (2*len(train_loader))

            
            if rank == 0:
                print(f"Train time: {(total_time/60):.5f} ")
                with open(f"benchmarks/n2v/{cluster_run_file_name}.csv", "a") as f:
                    f.write(f"{batch_size},{patch_size},{num_patches},{epoch_loss},{(total_time/60):.4f}\n")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()