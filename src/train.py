import hydra
from omegaconf import DictConfig, OmegaConf

from utils.build import load_dataset

@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg: DictConfig):
    train_loader, test_loader = load_dataset(batch_size=cfg.data.batches, num_workers=cfg.data.num_workers, patch_size=cfg.data.patch_size)
    

if __name__ == "__main__":
    main()