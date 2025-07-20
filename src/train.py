import hydra
from omegaconf import DictConfig, OmegaConf

from utils.build import load_dataset

@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg: DictConfig):
    dataset = load_dataset(batch_size=16, num_workers=0, patch_size=256 )
    print(len(dataset))


if __name__ == "__main__":
    main()