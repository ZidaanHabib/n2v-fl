import time
from datetime import datetime
import hydra
from omegaconf import DictConfig


from utils.build import load_dataset, load_distributed_dataset, setup_loss, setup_optimizer, test_step_synthetic, train_step, test_step, seed, set_device, train_step_synthetic
from models.unet import UNet

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import os

from pathlib import Path

@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg: DictConfig):

    # read environment variables
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"]) 
    local_rank = int(os.environ["LOCAL_RANK"])

    # do the global seeding for RNGs
    seed_value = 42
    seed(seed_value)

    # set device to cuda if available otherwise whatever other backend
    device = set_device()
    
    #instantiate model
    model = UNet(1,cfg.model.base_channels, cfg.model.depth, batch_norm=cfg.model.batch_norm).to(local_rank)

    has_ground_truth = cfg.data.has_ground_truth

    # set up paths for data access
    root = Path().resolve().parent
    data_dir = Path(cfg.data.data_dir) if cfg.data.data_dir else root / "data" / "preprocessed"

    if device == torch.device("cuda") and torch.cuda.device_count() > 1:
        torch.cuda.set_device(local_rank) # associate cuda device with particular gpu
        dist.init_process_group(backend='nccl', init_method="env://") # nccl is cuda backend for multi-gpu comms, env simply means environment
        
        # wrap model in DDP for distributed processing
        model = DDP(model,device_ids=[local_rank], output_device=local_rank)
        #load dataset:
        train_loader, test_loader = load_distributed_dataset(rank=rank, world_size=world_size, data_dir=data_dir, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, patch_size=cfg.data.patch_size, patches_per_image=cfg.data.patches_per_image, seed=seed_value, has_ground_truth=has_ground_truth)
    else:
        train_loader, test_loader = load_dataset(data_dir=data_dir, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, patch_size=cfg.data.patch_size, patches_per_image=cfg.data.patches_per_image, seed=seed_value)


    #set up loss and optimizer
    loss_fn = setup_loss()
    optim = setup_optimizer(model_params=model.parameters(), type=cfg.optimizer.type, lr=cfg.optimizer.lr, betas=cfg.optimizer.betas)

    # ensure directories exist for saving
    cluster_run_dir_name = f"cluster-run-{datetime.now().strftime("%d_%m_%Hh%M")}"
    if rank == 0:
        Path(f"runs/{cluster_run_dir_name}/checkpoints").mkdir(parents=True, exist_ok=True)
        Path(f"runs/{cluster_run_dir_name}/output").mkdir(parents=True, exist_ok=True)

    epochs = int(cfg.train.epochs)


    

    dist.barrier() # synchronise processes before starting timer
    start_time = time.perf_counter()
    if has_ground_truth:
        perform_synthetic_model_training(rank, device, model,train_loader, test_loader, loss_fn, optim, cluster_run_dir_name, epochs, start_time)
    else:
        perform_model_training(rank, device, model, train_loader, test_loader, loss_fn, optim, cluster_run_dir_name, epochs, start_time)

    dist.destroy_process_group()

def perform_model_training(rank, device, model, train_loader, test_loader, loss_fn, optim, cluster_run_dir_name, epochs, start_time):

    best_val_loss = torch.tensor(float("inf"), device=device, requires_grad=False)
    avg_train_losses = torch.zeros(epochs, device=device, requires_grad=False) 
    avg_test_losses =  torch.zeros(epochs, device=device, requires_grad=False)

    for epoch in range(1,epochs+1):
        if rank == 0:
            print(f"Epoch: {epoch}-----------------------------------------\n\n")
        
        avg_train_loss = train_step(data_loader=train_loader, 
            model=model, 
            loss_fn=loss_fn,
            opt=optim,
            device=device,
            epoch=epoch,
            rank=rank,
            dir_name=cluster_run_dir_name,
        )

        avg_test_loss = test_step(data_loader=test_loader,
            model=model,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            rank=rank,
            dir_name=cluster_run_dir_name,
        )
        
        # Reduce train loss across different processes
        dist.all_reduce(avg_train_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(avg_test_loss,op=dist.ReduceOp.AVG)


        if rank == 0:
            # append epoch loss to running list: 
            avg_train_losses[epoch - 1] = avg_train_loss
            # append epoch loss to running list:
            avg_test_losses[epoch - 1] = avg_test_loss
            
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optim.state_dict(),
                    "avg_train_loss": avg_train_loss,
                    "avg_test_loss":    avg_test_loss
                },
                f"runs/{cluster_run_dir_name}/checkpoints/last.pth",
            )

            # If it’s the best so far, also save “best.pth”
            if avg_test_loss < best_val_loss:
                best_val_loss = avg_test_loss
                torch.save(
                    {
                        "model_state": model.state_dict(),  # you can omit optimizer if only inference
                        "avg_test_loss": avg_test_loss,
                        "avg_train_loss": avg_train_loss
                    },
                    f"runs/{cluster_run_dir_name}/checkpoints/best.pth",
                )
        
    dist.barrier()
    total_time = time.perf_counter() - start_time
    if rank == 0:
        print(f"Training time: {(total_time / 60):.2f} minutes")
        print("Writing avg epoch losses to file...")
        with open(f"runs/{cluster_run_dir_name}/output/epoch_losses.txt", "w") as f:
            f.write("Train_loss,Test_loss\n")
            f.writelines(f"{train_loss.item()},{test_loss.item()}\n" for (train_loss, test_loss) in zip(avg_train_losses, avg_test_losses))
        print("Done")
    
def perform_synthetic_model_training(rank, device, model, train_loader, test_loader, loss_fn, optim, cluster_run_dir_name, epochs, start_time):
    #Initialise
    best_val_loss = torch.tensor(float("inf"), device=device, requires_grad=False)
    avg_train_losses = []
    avg_test_losses =  []
    avg_train_psnr_values = []
    avg_test_psnr_values = []

    for epoch in range(1,epochs+1):
        if rank == 0:
            print(f"Epoch: {epoch}-----------------------------------------\n\n")
        
        avg_train_loss, avg_train_psnr = train_step_synthetic(data_loader=train_loader, 
            model=model, 
            loss_fn=loss_fn,
            opt=optim,
            device=device,
            epoch=epoch,
            rank=rank,
            dir_name=cluster_run_dir_name,
        )

        avg_test_loss, avg_test_psnr = test_step_synthetic(data_loader=test_loader,
            model=model,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            rank=rank,
            dir_name=cluster_run_dir_name,
        )
        
        # Reduce train loss across different processes
        dist.all_reduce(avg_train_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(avg_test_loss,op=dist.ReduceOp.AVG)
        
        # Reduce psnr across different processes
        dist.all_reduce(avg_train_psnr, op=dist.ReduceOp.AVG)
        dist.all_reduce(avg_test_psnr, op=dist.ReduceOp.AVG)


        if rank == 0:
            # append epoch loss to running list: 
            avg_train_losses.append(avg_train_loss.item())
            # append epoch loss to running list:
            avg_test_losses.append(avg_test_loss.item())
            # append epoch psnr to running list:
            avg_train_psnr_values.append(avg_train_psnr.item())
            # append epoch psnr to running list:
            avg_test_psnr_values.append(avg_test_psnr.item())         
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optim.state_dict(),
                    "avg_train_loss": avg_train_loss,
                    "avg_test_loss":    avg_test_loss,
                    "avg_train_psnr": avg_train_psnr,
                    "avg_test_psnr": avg_test_psnr
                },
                f"runs/{cluster_run_dir_name}/checkpoints/last.pth",
            )

            # If it’s the best so far, also save “best.pth”
            if avg_test_loss < best_val_loss:
                best_val_loss = avg_test_loss
                torch.save(
                    {
                        "model_state": model.state_dict(),  # you can omit optimizer if only inference
                        "avg_test_loss": avg_test_loss,
                        "avg_train_loss": avg_train_loss,
                        "avg_train_psnr": avg_train_psnr,
                        "avg_test_psnr": avg_test_psnr
                    },
                    f"runs/{cluster_run_dir_name}/checkpoints/best.pth",
                )
        
    dist.barrier()
    total_time = time.perf_counter() - start_time
    if rank == 0:
        print(f"Training time: {(total_time / 60):.2f} minutes")
        print("Writing avg epoch losses to file...")
        with open(f"runs/{cluster_run_dir_name}/output/epoch_losses.txt", "w") as f:
            f.write("Train_loss,Test_loss\n")
            f.writelines(f"{train_loss},{test_loss}\n" for (train_loss, test_loss) in zip(avg_train_losses, avg_test_losses))
        print("Done")
        print("Writing avg epoch psnr values to file...")
        with open(f"runs/{cluster_run_dir_name}/output/epoch_psnr_values.txt", "w") as f:
            f.write("Train_psnr,Test_psnr\n")
            f.writelines(f"{train_psnr},{test_psnr}\n" for (train_psnr, test_psnr) in zip(avg_train_psnr_values, avg_test_psnr_values))
        print("Done")

if __name__ == "__main__":
    main()