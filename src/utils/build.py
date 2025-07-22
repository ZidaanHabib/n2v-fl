
from typing import Iterable, Optional, Tuple
import torch.nn as nn
import torch.optim 
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision.transforms import v2
from pathlib import Path

from data.dataset import N2NImageDataset

def set_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu" )
    print(f"Running on device: {device}")
    return device

def distribute_model(model: nn.Module, device: torch.device) -> nn.Module :
    if torch.cuda.device_count() > 1:
        print(f"→ Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    return model.to(device)

def load_dataset(batch_size: int, num_workers = 0, patch_size = 256 ):
    # define transforms for dataset
    transform_confocal = v2.Compose([
        v2.ToImage(),
        v2.ConvertImageDtype(torch.float32),
        v2.RandomCrop(size=patch_size,padding=512,padding_mode="reflect"),
    ])

    transform_nucleus = v2.Compose([
        v2.ToImage(),
        v2.ConvertImageDtype(torch.float32),
        v2.Pad(padding=256,padding_mode="reflect"),
        v2.Pad(padding=512,padding_mode="reflect"), 
        v2.RandomCrop(size=patch_size),
    ])

    transform = v2.Compose([
        v2.ToImage(),
        v2.ConvertImageDtype(torch.float32),
        v2.RandomCrop(size=patch_size),
    ])

    base_dir = Path().resolve().parent

    subdatasets = [ N2NImageDataset(base_dir, dataset="20x-noise1", subdataset="actin-20x-noise1",transform=transform, patches_per_image=16),
                N2NImageDataset(base_dir, dataset="20x-noise1", subdataset="mito-20x-noise1",transform=transform, patches_per_image=16),
                N2NImageDataset(base_dir, dataset="60x-noise1", subdataset="actin-60x-noise1",transform=transform, patches_per_image=16),
                N2NImageDataset(base_dir, dataset="60x-noise1", subdataset="mito-60x-noise1",transform=transform, patches_per_image=16),
                N2NImageDataset(base_dir, dataset="60x-noise2", subdataset="actin-60x-noise2",transform=transform, patches_per_image=16),
                N2NImageDataset(base_dir, dataset="60x-noise2", subdataset="mito-60x-noise2",transform=transform, patches_per_image=16),
                N2NImageDataset(base_dir, dataset="confocal", subdataset="actin-confocal",transform=transform_confocal, patches_per_image=16),
                N2NImageDataset(base_dir, dataset="confocal", subdataset="mito-confocal",transform=transform_confocal, patches_per_image=16),
                N2NImageDataset(base_dir, dataset="membrane", subdataset="membrane",transform=transform, patches_per_image=16),
                N2NImageDataset(base_dir, dataset="nucleus", subdataset="nucleus",transform=transform_nucleus, patches_per_image=16)
    ]
    dataset = ConcatDataset(subdatasets)
    dataset_length = len(dataset)
    train_size = int(0.8 * dataset_length)
    test_size = dataset_length - train_size

    train_dataset, test_dataset = random_split(dataset, (train_size,test_size))

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
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, test_loader

def setup_loss():
    return nn.MSELoss()

def setup_optimizer(model_params: Iterable[nn.Parameter], type: str, lr: float, betas: Optional[Tuple[float, float]] = None):
    if type == 'Adam':
        optim = torch.optim.Adam(params=model_params,betas=betas, lr=lr)
    elif type == "SGD":
        optim = torch.optim.SGD(params=model_params,lr=lr)
    else:
        raise ValueError("Invalid optimizer")
    
    return optim


def train_step(model, data_loader, loss_fn, opt, device, epoch):
    model.train()
    running_loss, running_psnr = 0.0, 0.0
    losses = []

    for batch, (X, y) in enumerate(data_loader):
        
        X, y = X.to(device), y.to(device)

        opt.zero_grad()
        denoised = model(X)
        loss   = loss_fn(denoised, y)

        loss.backward()
        opt.step()

        current_loss = loss.item()
        losses.append(current_loss)
        running_loss   += current_loss

        with torch.no_grad():
            # if your data range is [0,1]; otherwise pass max_val=255
            batch_psnr = compute_psnr(denoised, y, max_val=1.0)
        running_psnr += batch_psnr
        print(f"Batch {batch} done | Batch loss: {current_loss:.5f}")
 

    epoch_loss   = running_loss   / len(data_loader)
    epoch_psnr = running_psnr/ len(data_loader)

    print(f"Train loss: {epoch_loss:.5f} | PSNR: {epoch_psnr:.2f}")

    if epoch % 20 == 0 or epoch == 1:
        with open("output/train_losses.txt", "w") as f:
            f.writelines(f"{loss:.5f}\n" for loss in losses)


def test_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    epoch: int
):
    # switch to eval mode
    model.eval()

    total_loss = 0.0
    total_psnr  = 0.0
    losses = []

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
            losses.append(current_loss)
            total_loss += current_loss

            # PSNR
            batch_psnr = compute_psnr(denoised, y, max_val=1.0)
            total_psnr += batch_psnr

            print(f"Validation Batch {batch} done | Validation batch loss: {current_loss:.5f} ")

    if epoch % 20 == 0 or epoch == 1:
        with open("output/test_losses.txt", "w") as f:
            f.writelines(f"{loss:.5f}\n" for loss in losses)

    # Average over batches
    avg_loss = total_loss / len(data_loader)
    avg_psnr  = total_psnr  / len(data_loader)

    print(f"Test loss: {avg_loss:.5f} | Test PSNR: {avg_psnr:.2f}\n")
    return avg_loss, avg_psnr

def compute_psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    mse = torch.clamp(mse, min=1e-10) # prevent log(infinity) issues
    return 10 * torch.log10(max_val**2 / mse)

def seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
