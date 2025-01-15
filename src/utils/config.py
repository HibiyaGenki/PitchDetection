import omegaconf


def load_config(config_path: str) -> omegaconf.DictConfig:
    return omegaconf.OmegaConf.load(config_path)


def save_config(config: omegaconf.DictConfig, save_path: str) -> None:
    omegaconf.OmegaConf.save(config, save_path)
