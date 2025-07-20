
from torch.nn import functional as F    
from torch.utils.data import DataLoader, Dataset, ConcatDataset
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
        v2.RandomCrop(size=patch_size,padding=768,padding_mode="reflect"),
    ])

    transform = v2.Compose([
        v2.ToImage(),
        v2.RandomCrop(size=patch_size),
    ])

    base_dir = Path().resolve().parent
    print(base_dir)
    # tst = N2NImageDataset(base_dir, dataset="20x-noise1", subdataset="actin-20x-noise1",transform=transform, patches_per_image=256)

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
    return ConcatDataset(subdatasets)