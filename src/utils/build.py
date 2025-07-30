
from typing import Iterable, Tuple
import torch.nn as nn
import torch.optim 
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset, random_split, DistributedSampler
from torchvision.transforms import v2
from pathlib import Path

from data.dataset import N2NImageDataset, N2NSyntheticImageDataset

# define transforms for dataset
transform_confocal = v2.Compose([
    v2.ToImage(),
    v2.ConvertImageDtype(torch.float32,),
    v2.Pad(padding=512, padding_mode='reflect'),
])

transform_nucleus = v2.Compose([
    v2.ToImage(),
    v2.ConvertImageDtype(torch.float32),
    v2.Pad(padding=256,padding_mode="reflect"),
    v2.Pad(padding=512,padding_mode="reflect"), 
])

transform = v2.Compose([
    v2.ToImage(),
    v2.ConvertImageDtype(torch.float32),
])
transform_confocal_rot = v2.Compose([
    v2.ToImage(),
    v2.RandomRotation((90,90)),
    v2.ConvertImageDtype(torch.float32,),
    v2.Pad(padding=512, padding_mode='reflect'),
])

transform_nucleus_rot = v2.Compose([
    v2.ToImage(),
    v2.RandomRotation((90,90)),
    v2.ConvertImageDtype(torch.float32),
    v2.Pad(padding=256,padding_mode="reflect"),
    v2.Pad(padding=512,padding_mode="reflect"), 
    
])

transform_rot = v2.Compose([
    v2.ToImage(),
    v2.RandomRotation((90,90)),
    v2.ConvertImageDtype(torch.float32),
])

def set_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu" )
    print(f"Running on device: {device}")
    return device

def distribute_model(model: nn.Module, device: torch.device) -> nn.Module :
    if torch.cuda.device_count() > 1:
        print(f"→ Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    return model.to(device)

def load_dataset(data_dir: Path, batch_size: int, num_workers: int = 0, patch_size: int = 256, patches_per_image: int = 64, seed: int = 42) -> Tuple[DataLoader, DataLoader]:
   
    dataset =  create_dataset(data_dir, patch_size, patches_per_image)
    dataset_length = len(dataset)
    train_size = int(0.9 * dataset_length)
    test_size = dataset_length - train_size

    gen = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = random_split(dataset, (train_size,test_size), generator=gen)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, test_loader

def  create_dataset(data_dir, patch_size, patches_per_image) :
    subdatasets = [ N2NImageDataset(data_dir, dataset="20x-noise1", subdataset="actin-20x-noise1",transform=transform, patch_size=patch_size, patches_per_image=patches_per_image),
                N2NImageDataset(data_dir, dataset="20x-noise1", subdataset="mito-20x-noise1",transform=transform, patch_size=patch_size, patches_per_image=patches_per_image),
                N2NImageDataset(data_dir, dataset="60x-noise1", subdataset="actin-60x-noise1",transform=transform, patch_size=patch_size, patches_per_image=patches_per_image),
                N2NImageDataset(data_dir, dataset="60x-noise1", subdataset="mito-60x-noise1",transform=transform, patch_size=patch_size, patches_per_image=patches_per_image),
                N2NImageDataset(data_dir, dataset="60x-noise2", subdataset="actin-60x-noise2",transform=transform, patch_size=patch_size, patches_per_image=patches_per_image),
                N2NImageDataset(data_dir, dataset="60x-noise2", subdataset="mito-60x-noise2",transform=transform, patch_size=patch_size, patches_per_image=patches_per_image),
                N2NImageDataset(data_dir, dataset="confocal", subdataset="actin-confocal",transform=transform_confocal, patch_size=patch_size, patches_per_image=patches_per_image),
                N2NImageDataset(data_dir, dataset="confocal", subdataset="mito-confocal",transform=transform_confocal, patch_size=patch_size, patches_per_image=patches_per_image),
                N2NImageDataset(data_dir, dataset="membrane", subdataset="membrane",transform=transform, patch_size=patch_size, patches_per_image=patches_per_image),
                N2NImageDataset(data_dir, dataset="nucleus", subdataset="nucleus",transform=transform_nucleus, patch_size=patch_size, patches_per_image=patches_per_image)
                # N2NImageDataset(data_dir, dataset="20x-noise1", subdataset="actin-20x-noise1",transform=transform_rot, patch_size=patch_size, patches_per_image=patches_per_image),
                # N2NImageDataset(data_dir, dataset="20x-noise1", subdataset="mito-20x-noise1",transform=transform_rot, patch_size=patch_size, patches_per_image=patches_per_image),
                # N2NImageDataset(data_dir, dataset="60x-noise1", subdataset="actin-60x-noise1",transform=transform_rot, patch_size=patch_size, patches_per_image=patches_per_image),
                # N2NImageDataset(data_dir, dataset="60x-noise1", subdataset="mito-60x-noise1",transform=transform_rot, patch_size=patch_size, patches_per_image=patches_per_image),
                # N2NImageDataset(data_dir, dataset="60x-noise2", subdataset="actin-60x-noise2",transform=transform_rot, patch_size=patch_size, patches_per_image=patches_per_image),
                # N2NImageDataset(data_dir, dataset="60x-noise2", subdataset="mito-60x-noise2",transform=transform_rot, patch_size=patch_size, patches_per_image=patches_per_image),
                # N2NImageDataset(data_dir, dataset="confocal", subdataset="actin-confocal",transform=transform_confocal_rot, patch_size=patch_size, patches_per_image=patches_per_image),
                # N2NImageDataset(data_dir, dataset="confocal", subdataset="mito-confocal",transform=transform_confocal_rot, patch_size=patch_size, patches_per_image=patches_per_image),
                # N2NImageDataset(data_dir, dataset="membrane", subdataset="membrane",transform=transform_rot, patch_size=patch_size, patches_per_image=patches_per_image),
                # N2NImageDataset(data_dir, dataset="nucleus", subdataset="nucleus",transform=transform_nucleus_rot, patch_size=patch_size, patches_per_image=patches_per_image)
    ]
    dataset = ConcatDataset(subdatasets)
    return dataset

def  create_synthetic_dataset(data_dir, patch_size, patches_per_image) :
    subdatasets = [ N2NSyntheticImageDataset(data_dir, dataset="20x-noise1", subdataset="actin-20x-noise1",transform=transform, patch_size=patch_size, patches_per_image=patches_per_image),
                N2NSyntheticImageDataset(data_dir, dataset="20x-noise1", subdataset="mito-20x-noise1",transform=transform, patch_size=patch_size, patches_per_image=patches_per_image),
                N2NSyntheticImageDataset(data_dir, dataset="60x-noise1", subdataset="actin-60x-noise1",transform=transform, patch_size=patch_size, patches_per_image=patches_per_image),
                N2NSyntheticImageDataset(data_dir, dataset="60x-noise1", subdataset="mito-60x-noise1",transform=transform, patch_size=patch_size, patches_per_image=patches_per_image),
                N2NSyntheticImageDataset(data_dir, dataset="60x-noise2", subdataset="actin-60x-noise2",transform=transform, patch_size=patch_size, patches_per_image=patches_per_image),
                N2NSyntheticImageDataset(data_dir, dataset="60x-noise2", subdataset="mito-60x-noise2",transform=transform, patch_size=patch_size, patches_per_image=patches_per_image),
                N2NSyntheticImageDataset(data_dir, dataset="confocal", subdataset="actin-confocal",transform=transform_confocal, patch_size=patch_size, patches_per_image=patches_per_image),
                N2NSyntheticImageDataset(data_dir, dataset="confocal", subdataset="mito-confocal",transform=transform_confocal, patch_size=patch_size, patches_per_image=patches_per_image),
                N2NSyntheticImageDataset(data_dir, dataset="membrane", subdataset="membrane",transform=transform, patch_size=patch_size, patches_per_image=patches_per_image),
                N2NSyntheticImageDataset(data_dir, dataset="nucleus", subdataset="nucleus",transform=transform_nucleus, patch_size=patch_size, patches_per_image=patches_per_image)
    ]
    dataset = ConcatDataset(subdatasets)
    return dataset

def load_distributed_dataset(world_size: int, rank: int, data_dir: Path, batch_size: int, num_workers: int, patch_size: int, patches_per_image: int, seed: int = 42, has_ground_truth: bool = False):
    dataset = create_synthetic_dataset(data_dir,patch_size, patches_per_image) if has_ground_truth else create_dataset(data_dir, patch_size, patches_per_image)
    dataset_length = len(dataset)
    train_size = int(0.9 * dataset_length)
    test_size = dataset_length - train_size
    
    gen = torch.Generator().manual_seed(seed)

    train_dataset, test_dataset = random_split(dataset, (train_size,test_size), gen)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, test_loader

def setup_loss():
    return nn.MSELoss()

def setup_optimizer(model_params: Iterable[nn.Parameter], type: str, lr: float, betas: Tuple[float, float] = (0.9, 0.999)):
    if type == 'Adam':
        optim = torch.optim.Adam(params=model_params,betas=betas, lr=lr)
    elif type == "SGD":
        optim = torch.optim.SGD(params=model_params,lr=lr)
    else:
        raise ValueError("Invalid optimizer")
    
    return optim


def train_step(model, data_loader, loss_fn, opt, device, epoch, rank, dir_name: str) -> torch.Tensor :
    model.train()
    running_loss = torch.tensor(0, dtype=torch.float32, device=device, requires_grad=False)
    losses = torch.zeros(len(data_loader),device=device,requires_grad=False) 


    for batch, (X, y) in enumerate(data_loader):
        
        X, y = X.to(device), y.to(device)

        opt.zero_grad()
        denoised = model(X)
        loss   = loss_fn(denoised, y)

        loss.backward()
        opt.step()

        current_loss = loss.item()
        losses[batch] = current_loss
        running_loss   += current_loss

        print(f"Rank {rank} | Batch {batch} done | Batch loss: {current_loss:.5f}")
    

    dist.all_reduce(losses, op=dist.ReduceOp.AVG)
    if rank == 0:
        if epoch % 20 == 0 or epoch == 1:
            with open(f"runs/{dir_name}/output/epoch_{epoch}_train_losses.txt", "w") as f:
                f.write(f"Epoch {epoch}\n")
                f.writelines(f"{loss.item():.5f}\n" for loss in losses)

    rank_epoch_loss   = running_loss   / len(data_loader)
    # epoch_psnr = running_psnr/ len(data_loader)

    print(f"Rank {rank} | Train loss: {rank_epoch_loss:.5f} ")
    return rank_epoch_loss

def train_step_synthetic(model, data_loader, loss_fn, opt, device, epoch, rank, dir_name: str) -> Tuple[torch.Tensor, torch.Tensor] :
    model.train()
    running_loss = torch.tensor(0, dtype=torch.float32, device=device, requires_grad=False)
    losses = torch.zeros(len(data_loader),device=device,requires_grad=False) 
    running_psnr = torch.tensor(0, dtype=torch.float32, device=device, requires_grad=False)
    psnr_list = torch.zeros(len(data_loader), device=device, requires_grad=False)


    for batch, (X, y, clean) in enumerate(data_loader):
        
        X, y, clean = X.to(device), y.to(device), clean.to(device)

        opt.zero_grad()
        denoised = model(X)
        loss   = loss_fn(denoised, y)

        loss.backward()
        opt.step()

        current_loss = loss.item()
        losses[batch] = current_loss
        running_loss   += current_loss

        with torch.inference_mode():
            current_psnr = compute_psnr(denoised, y, clean, 1.0).item()
        psnr_list[batch] = current_psnr
        running_psnr += current_psnr

        print(f"Rank {rank} | Batch {batch} done | Batch loss: {current_loss:.5f} | Batch PSNR: {current_psnr}")
    

    dist.all_reduce(losses, op=dist.ReduceOp.AVG)
    dist.all_reduce(psnr_list, op=dist.ReduceOp.AVG)
    if rank == 0:
        if epoch % 20 == 0 or epoch == 1:
            with open(f"runs/{dir_name}/output/epoch_{epoch}_train_losses.txt", "w") as f:
                f.write(f"Epoch {epoch}\n")
                f.writelines(f"{loss.item():.5f}\n" for loss in losses)
            with open(f"runs/{dir_name}/output/epoch_{epoch}_train_psnr.txt", "w") as f:
                f.write(f"Epoch {epoch}\n")
                f.writelines(f"{psnr.item():.2f}\n" for psnr in psnr_list)

    rank_epoch_loss   = running_loss   / len(data_loader)
    rank_epoch_psnr = running_psnr/ len(data_loader)

    print(f"Rank {rank} | Epoch {epoch} | Avg Epoch Train loss: {rank_epoch_loss:.5f} | Avg Epoch Train PSNR: {rank_epoch_psnr:.2f} ")
    return rank_epoch_loss, rank_epoch_psnr

def test_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    epoch: int,
    rank: int,
    dir_name: str
) -> torch.Tensor :
    # switch to eval mode
    model.eval()

    total_loss = torch.tensor(0, dtype=torch.float32, device=device, requires_grad=False)
    # total_psnr  = 0.0
    losses = torch.zeros(len(data_loader), device=device, requires_grad=False)

    # No gradients needed
    with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):
            # Send data to the same device
            X, y = X.to(device), y.to(device)

            # Forward pass
            denoised = model(X)

            # Compute loss & metric (use .item() to get floats)
            loss = loss_fn(denoised, y)
            current_loss = loss.item()
            losses[batch] = current_loss
            total_loss += current_loss

            # # PSNR - ignore for now
            # batch_psnr = compute_psnr(denoised, y, max_val=1.0)
            # total_psnr += batch_psnr

            print(f"Rank {rank} | Validation Batch {batch} done | Validation batch loss: {current_loss:.5f} ")

    dist.all_reduce(losses, op=dist.ReduceOp.AVG)
    if epoch % 20 == 0 or epoch == 1:
        with open(f"runs/{dir_name}/output/epoch_{epoch}_test_losses.txt", "w") as f:
            f.write(f"Epoch {epoch}\n")
            f.writelines(f"{loss.item():.5f}\n" for loss in losses)

    # Average over batches
    rank_avg_loss = total_loss / len(data_loader)
    # avg_psnr  = total_psnr  / len(data_loader)

    print(f"Rank {rank} | Test loss: {rank_avg_loss:.5f} ")
    return rank_avg_loss

def test_step_synthetic(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    epoch: int,
    rank: int,
    dir_name: str
) -> Tuple[torch.Tensor, torch.Tensor] :
    # switch to eval mode
    model.eval()

    total_loss = torch.tensor(0, dtype=torch.float32, device=device, requires_grad=False)
    total_psnr  = torch.tensor(0, dtype=torch.float32, device=device, requires_grad=False)
    losses = torch.zeros(len(data_loader), device=device, requires_grad=False)
    psnr_list = torch.zeros(len(data_loader), device=device, requires_grad=False)
    # No gradients needed
    with torch.inference_mode():
        for batch, (X, y, clean) in enumerate(data_loader):
            # Send data to the same device
            X, y = X.to(device), y.to(device)

            # Forward pass
            denoised = model(X)

            # Compute loss & metric (use .item() to get floats)
            loss = loss_fn(denoised, y)
            current_loss = loss.item()
            losses[batch] = current_loss
            total_loss += current_loss

            batch_psnr = compute_psnr(denoised, y, clean, max_val=1.0).item()
            psnr_list[batch] = batch_psnr
            total_psnr += batch_psnr

            print(f"Rank {rank} | Validation Batch {batch} done | Validation batch loss: {current_loss:.5f} | PSNR: {batch_psnr}")

    dist.all_reduce(losses, op=dist.ReduceOp.AVG)
    dist.all_reduce(psnr_list, op=dist.ReduceOp.AVG)
    if rank  == 0:
        if epoch % 20 == 0 or epoch == 1:
            with open(f"runs/{dir_name}/output/epoch_{epoch}_test_losses.txt", "w") as f:
                f.write(f"Epoch {epoch}\n")
                f.writelines(f"{loss.item():.5f}\n" for loss in losses)
            with open(f"runs/{dir_name}/output/epoch_{epoch}_test_psnr.txt", "w") as f:
                f.write(f"Epoch {epoch}\n")
                f.writelines(f"{psnr.item():.2f}\n" for psnr in psnr_list)

    # Average over batches
    rank_avg_loss = total_loss / len(data_loader)
    rank_avg_psnr = total_psnr / len(data_loader)

    print(f"Rank {rank}, Epoch {epoch} | Avg Epoch Test loss: {rank_avg_loss:.5f} | Avg Epoch Test PSNR: {rank_avg_psnr} ")
    return rank_avg_loss, rank_avg_psnr

def compute_psnr(pred, target, clean, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    mse = torch.clamp(mse, min=1e-10) # prevent log(infinity) issues
    return 10 * torch.log10(max_val**2 / mse)

def seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
