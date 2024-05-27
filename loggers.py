from lightning.pytorch.loggers import WandbLogger


def get_wandb_logger(config):
    wandb_logger = WandbLogger(
        project=config.project, entity=config.entity, save_dir=config.save_dir
    )
    
    return wandb_logger
