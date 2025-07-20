
from torch.nn import functional as F    
from torch.utils.data import  Dataset

from torchvision import io
from torchvision.transforms import v2

from pathlib import Path


# define custom image dataset for N2n
class N2NImageDataset(Dataset):
    def __init__(self,root: Path, dataset: str, subdataset: str, transform, patches_per_image: int):
        super().__init__()
        self.root = root
        dataset_dir = root / "data" / "preprocessed" / dataset
        self.image_prefix = f"{dataset}-{subdataset}-lowsnr" 
        self.target_prefix = f"{dataset}-{subdataset}-highsnr" 
        self.image_dir = dataset_dir / self.image_prefix # treat lowsnr as input image
        self.target_dir = dataset_dir / self.target_prefix # treat highsnr as target image
        self.transform = transform
        self.patches_per_image = patches_per_image
    
    def __len__(self):
        return sum(self.patches_per_image for path in self.image_dir.iterdir() if path.is_file()) # count number of images in directory
    
    def __getitem__(self, index):
        img_index = (index // self.patches_per_image ) 
        image_patch = self.transform(io.decode_image(self.image_dir / f"{self.image_prefix}-{img_index}.png", mode=io.ImageReadMode.GRAY))
        target_patch = self.transform(io.decode_image(self.target_dir / f"{self.target_prefix}-{img_index}.png", mode=io.ImageReadMode.GRAY))
        return image_patch, target_patch

