
from torch.nn import functional as F    
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision.transforms import v2
from pathlib import Path

from data.dataset import N2NImageDataset


def load_dataset(batch_size: int, num_workers = 0, patch_size = 256 ):
    # define transforms for dataset
    transform_confocal = v2.Compose([
        v2.ToImage(),
        v2.RandomCrop(size=patch_size,padding=512,padding_mode="reflect"),
    ])

    transform_nucleus = v2.Compose([
        v2.ToImage(),
        v2.Pad(padding=768,padding_mode="reflect"),
        v2.RandomCrop(size=patch_size),
    ])

    transform = v2.Compose([
        v2.ToImage(),
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
