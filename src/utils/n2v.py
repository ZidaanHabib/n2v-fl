from pathlib import Path
from typing import Tuple
import torch
import torch.distributed as dist
from data.dataset import N2VImageDataset, N2VSyntheticImageDataset
from torch.utils.data import DataLoader, ConcatDataset, random_split, DistributedSampler
from utils.build import compute_psnr, transform, transform_confocal, transform_nucleus


def  create_dataset(data_dir, patch_size, patches_per_image) :
    subdatasets = [ N2VImageDataset(data_dir, dataset="20x-noise1", subdataset="actin-20x-noise1",transform=transform, patch_size=patch_size, patches_per_image=patches_per_image),
                N2VImageDataset(data_dir, dataset="20x-noise1", subdataset="mito-20x-noise1",transform=transform, patch_size=patch_size, patches_per_image=patches_per_image),
                N2VImageDataset(data_dir, dataset="60x-noise1", subdataset="actin-60x-noise1",transform=transform, patch_size=patch_size, patches_per_image=patches_per_image),
                N2VImageDataset(data_dir, dataset="60x-noise1", subdataset="mito-60x-noise1",transform=transform, patch_size=patch_size, patches_per_image=patches_per_image),
                N2VImageDataset(data_dir, dataset="60x-noise2", subdataset="actin-60x-noise2",transform=transform, patch_size=patch_size, patches_per_image=patches_per_image),
                N2VImageDataset(data_dir, dataset="60x-noise2", subdataset="mito-60x-noise2",transform=transform, patch_size=patch_size, patches_per_image=patches_per_image),
                N2VImageDataset(data_dir, dataset="confocal", subdataset="actin-confocal",transform=transform_confocal, patch_size=patch_size, patches_per_image=patches_per_image),
                N2VImageDataset(data_dir, dataset="confocal", subdataset="mito-confocal",transform=transform_confocal, patch_size=patch_size, patches_per_image=patches_per_image),
                N2VImageDataset(data_dir, dataset="membrane", subdataset="membrane",transform=transform, patch_size=patch_size, patches_per_image=patches_per_image),
                N2VImageDataset(data_dir, dataset="nucleus", subdataset="nucleus",transform=transform_nucleus, patch_size=patch_size, patches_per_image=patches_per_image)
                # N2VImageDataset(data_dir, dataset="20x-noise1", subdataset="actin-20x-noise1",transform=transform_rot, patch_size=patch_size, patches_per_image=patches_per_image),
                # N2VImageDataset(data_dir, dataset="20x-noise1", subdataset="mito-20x-noise1",transform=transform_rot, patch_size=patch_size, patches_per_image=patches_per_image),
                # N2VImageDataset(data_dir, dataset="60x-noise1", subdataset="actin-60x-noise1",transform=transform_rot, patch_size=patch_size, patches_per_image=patches_per_image),
                # N2VImageDataset(data_dir, dataset="60x-noise1", subdataset="mito-60x-noise1",transform=transform_rot, patch_size=patch_size, patches_per_image=patches_per_image),
                # N2VImageDataset(data_dir, dataset="60x-noise2", subdataset="actin-60x-noise2",transform=transform_rot, patch_size=patch_size, patches_per_image=patches_per_image),
                # N2VImageDataset(data_dir, dataset="60x-noise2", subdataset="mito-60x-noise2",transform=transform_rot, patch_size=patch_size, patches_per_image=patches_per_image),
                # N2VImageDataset(data_dir, dataset="confocal", subdataset="actin-confocal",transform=transform_confocal_rot, patch_size=patch_size, patches_per_image=patches_per_image),
                # N2VImageDataset(data_dir, dataset="confocal", subdataset="mito-confocal",transform=transform_confocal_rot, patch_size=patch_size, patches_per_image=patches_per_image),
                # N2VImageDataset(data_dir, dataset="membrane", subdataset="membrane",transform=transform_rot, patch_size=patch_size, patches_per_image=patches_per_image),
                # N2VImageDataset(data_dir, dataset="nucleus", subdataset="nucleus",transform=transform_nucleus_rot, patch_size=patch_size, patches_per_image=patches_per_image)
    ]
    dataset = ConcatDataset(subdatasets)
    return dataset

def  create_synthetic_dataset(data_dir, patch_size, patches_per_image) :
    subdatasets = [ N2VSyntheticImageDataset(data_dir, dataset="20x-noise1", subdataset="actin-20x-noise1",transform=transform, patch_size=patch_size, patches_per_image=patches_per_image),
                N2VSyntheticImageDataset(data_dir, dataset="20x-noise1", subdataset="mito-20x-noise1",transform=transform, patch_size=patch_size, patches_per_image=patches_per_image),
                N2VSyntheticImageDataset(data_dir, dataset="60x-noise1", subdataset="actin-60x-noise1",transform=transform, patch_size=patch_size, patches_per_image=patches_per_image),
                N2VSyntheticImageDataset(data_dir, dataset="60x-noise1", subdataset="mito-60x-noise1",transform=transform, patch_size=patch_size, patches_per_image=patches_per_image),
                N2VSyntheticImageDataset(data_dir, dataset="60x-noise2", subdataset="actin-60x-noise2",transform=transform, patch_size=patch_size, patches_per_image=patches_per_image),
                N2VSyntheticImageDataset(data_dir, dataset="60x-noise2", subdataset="mito-60x-noise2",transform=transform, patch_size=patch_size, patches_per_image=patches_per_image),
                N2VSyntheticImageDataset(data_dir, dataset="confocal", subdataset="actin-confocal",transform=transform_confocal, patch_size=patch_size, patches_per_image=patches_per_image),
                N2VSyntheticImageDataset(data_dir, dataset="confocal", subdataset="mito-confocal",transform=transform_confocal, patch_size=patch_size, patches_per_image=patches_per_image),
                N2VSyntheticImageDataset(data_dir, dataset="membrane", subdataset="membrane",transform=transform, patch_size=patch_size, patches_per_image=patches_per_image),
                N2VSyntheticImageDataset(data_dir, dataset="nucleus", subdataset="nucleus",transform=transform_nucleus, patch_size=patch_size, patches_per_image=patches_per_image)
    ]
    dataset = ConcatDataset(subdatasets)
    return dataset

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
        drop_last=True
    )
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=True
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


def mask_batch(patches: torch.Tensor, num_masks: int) :
    device = patches.device
    B, C, H, W = patches.shape
    total = H * W

    corrupted = patches.clone()
    mask = torch.zeros((B,1,H,W), dtype=torch.bool, device=device)

    # flatten images for easier indexing
    patches_flat = patches.view(B,C, total)

    # get indices of pixels to mask randomly:
    noise = torch.rand(B, total, device=device) # shape (B,1)
    target_flat = torch.argsort(noise, dim=1)[:, :num_masks] # randomly select num_mask indices for each iamge in the batch 

    #replace each target with random pixel 
    rand = torch.randint(0, total-1, (B, num_masks), device=device)
    neighbour_flat = rand + (rand >= target_flat).to(torch.long) # (B, num_masks)

    neighbour_flat_exp = neighbour_flat.unsqueeze(1) # adding channel dimension (B,1, num_masks)
    neighbour_vals = torch.gather(input=patches_flat, dim=2, index=neighbour_flat_exp)

    # send masked values into associate positions

    target_flat_exp = target_flat.unsqueeze(1)
    corrupted_flat = corrupted.view(B,C,total)
    corrupted_flat = corrupted_flat.clone()
    corrupted_flat.scatter(2, target_flat_exp, neighbour_vals)
    
    # Build mask
    mask_flat = torch.zeros((B, total), dtype=torch.bool, device=device)
    # compute linear indices per batch and set True
    batch_idx = torch.arange(B, device=device).unsqueeze(1)  # (B,1)
    mask_flat[batch_idx, target_flat] = True  # (B, total)
    mask = mask_flat.view(B, 1, H, W)

    return patches, corrupted, mask  # target is original


def train_step(model, data_loader, loss_fn, opt, device, epoch, rank, dir_name: str, num_masks: int) -> Tuple[torch.Tensor, torch.Tensor] :
    model.train()
    running_loss = torch.tensor(0, dtype=torch.float32, device=device, requires_grad=False)
    losses = torch.zeros(len(data_loader),device=device,requires_grad=False) 
    running_psnr = torch.tensor(0, dtype=torch.float32, device=device, requires_grad=False)
    psnr_list = torch.zeros(len(data_loader), device=device, requires_grad=False)
    
    for batch_idx, (patches, clean) in enumerate(data_loader):

        # perform masking
        X, y, mask = mask_batch(patches, num_masks)
        
        X, y, mask = X.to(device), y.to(device), mask.to(device)

        opt.zero_grad()
        denoised = model(X)
        loss   = loss_fn(denoised[mask], y[mask])

        loss.backward()
        opt.step()

        current_loss = loss.item()
        losses[batch_idx] = current_loss
        running_loss   += current_loss

        with torch.inference_mode():
            current_psnr = compute_psnr(denoised, clean, 1.0).item()
        psnr_list[batch_idx] = current_psnr
        running_psnr += current_psnr


        print(f"Epoch {epoch} | Rank {rank} | Batch {batch_idx} done | Batch loss: {current_loss:.5f} | Batch PSNR: {current_psnr}")
    

    dist.all_reduce(losses, op=dist.ReduceOp.AVG)
    dist.all_reduce(psnr_list, op=dist.ReduceOp.AVG)
    if rank == 0:
        if epoch % 20 == 0 or epoch == 1:
            with open(f"runs/n2v/{dir_name}/output/epoch_{epoch}_train_losses.txt", "w") as f:
                f.write(f"Epoch {epoch}\n")
                f.writelines(f"{loss.item():.5f}\n" for loss in losses)
            with open(f"runs/n2v/{dir_name}/output/epoch_{epoch}_train_psnr.txt", "w") as f:
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
    dir_name: str, 
    num_masks: int
) -> Tuple[torch.Tensor, torch.Tensor] :
    # switch to eval mode
    model.eval()

    total_loss = torch.tensor(0, dtype=torch.float32, device=device, requires_grad=False)
    losses = torch.zeros(len(data_loader), device=device, requires_grad=False)
    psnr_list = torch.zeros(len(data_loader), device=device, requires_grad=False)
    total_psnr  = torch.tensor(0, dtype=torch.float32, device=device, requires_grad=False)
    # No gradients needed
    with torch.inference_mode():
        for batch, (patches, clean) in enumerate(data_loader):

            # perform masking
            X, y, mask = mask_batch(patches, num_masks)

            # Send data to the same device
            X, y, mask = X.to(device), y.to(device), mask.to(device)

            # Forward pass
            denoised = model(X)

            # Compute loss & metric (use .item() to get floats)
            loss = loss_fn(denoised[mask], y[mask])
            current_loss = loss.item()
            losses[batch] = current_loss
            total_loss += current_loss

            batch_psnr = compute_psnr(denoised, clean, max_val=1.0).item()
            psnr_list[batch] = batch_psnr
            total_psnr += batch_psnr

            print(f" Epoch {epoch} | Rank {rank} | Validation Batch {batch} done | Validation batch loss: {current_loss:.5f} | PSNR: {batch_psnr}")

    dist.all_reduce(losses, op=dist.ReduceOp.AVG)
    dist.all_reduce(psnr_list, op=dist.ReduceOp.AVG)
    if rank  == 0:
        if epoch % 20 == 0 or epoch == 1:
            with open(f"runs/n2v/{dir_name}/output/epoch_{epoch}_test_losses.txt", "w") as f:
                f.write(f"Epoch {epoch}\n")
                f.writelines(f"{loss.item():.5f}\n" for loss in losses)
            with open(f"runs/n2v/{dir_name}/output/epoch_{epoch}_test_psnr.txt", "w") as f:
                f.write(f"Epoch {epoch}\n")
                f.writelines(f"{psnr.item():.2f}\n" for psnr in psnr_list)

    # Average over batches
    rank_avg_loss = total_loss / len(data_loader)
    rank_avg_psnr = total_psnr / len(data_loader)

    print(f"Rank {rank}, Epoch {epoch} | Avg Epoch Test loss: {rank_avg_loss:.5f} | Avg Epoch Test PSNR: {rank_avg_psnr} ")
    return rank_avg_loss, rank_avg_psnr