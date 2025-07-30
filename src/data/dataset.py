
from torch.utils.data import  Dataset

from torchvision import io
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

from pathlib import Path


# define custom image dataset for N2n
class N2NImageDataset(Dataset):
    def __init__(self, data_dir: Path, dataset: str, subdataset: str, transform, patch_size: int, patches_per_image: int):
        super().__init__()
        dataset_dir = data_dir / dataset
        self.image_prefix = f"{dataset}-{subdataset}-lowsnr" 
        self.target_prefix = f"{dataset}-{subdataset}-highsnr" 
        self.image_dir = dataset_dir / self.image_prefix # treat lowsnr as input image
        self.target_dir = dataset_dir / self.target_prefix # treat highsnr as target image
        self.transform = transform
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
    
    def __len__(self):
        return sum(self.patches_per_image for path in self.image_dir.iterdir() if path.is_file()) # count number of images in directory
    
    def __getitem__(self, index):
        img_index = (index // self.patches_per_image )
        image = self.transform(io.decode_image(self.image_dir / f"{self.image_prefix}-{img_index}.png", mode=io.ImageReadMode.GRAY))
        target = self.transform(io.decode_image(self.target_dir / f"{self.target_prefix}-{img_index}.png", mode=io.ImageReadMode.GRAY))
        top, left, height, width = v2.RandomCrop(self.patch_size).get_params(image,(self.patch_size,self.patch_size))
        image_patch = F.crop(image, top, left, height, width)
        target_patch = F.crop(target, top, left, height, width)
        return image_patch, target_patch
    

# define custom image dataset for Synthetic data n2n
class N2NSyntheticImageDataset(Dataset):
    def __init__(self, data_dir: Path, dataset: str, subdataset: str, transform, patch_size: int, patches_per_image: int):
        super().__init__()
        dataset_dir = data_dir / dataset
        self.subdataset = subdataset
        self.image_dir = dataset_dir / self.subdataset # treat lowsnr as input image
        self.transform = transform
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
    
    def __len__(self):
        return sum(self.patches_per_image for path in self.image_dir.iterdir() if path.is_file() and path.name.endswith("-clean.png")) # count number of images in directory
    
    def __getitem__(self, index):
        img_index = (index // self.patches_per_image )
        image = self.transform(io.decode_image(self.image_dir / f"{self.subdataset}-noisy-first-{img_index}.png", mode=io.ImageReadMode.GRAY))
        target = self.transform(io.decode_image(self.image_dir/ f"{self.subdataset}-noisy-second-{img_index}.png", mode=io.ImageReadMode.GRAY))
        clean = self.transform(io.decode_image(self.image_dir/ f"{self.subdataset}-clean-{img_index}.png", mode=io.ImageReadMode.GRAY))
        top, left, height, width = v2.RandomCrop(self.patch_size).get_params(image,(self.patch_size,self.patch_size))
        image_patch = F.crop(image, top, left, height, width)
        target_patch = F.crop(target, top, left, height, width)
        clean_patch = F.crop(clean, top, left, height, width)
        return image_patch, target_patch, clean_patch
