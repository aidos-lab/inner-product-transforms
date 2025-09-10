import pydantic
from typing import Any
from lightning.pytorch.loggers import WandbLogger


class LogConfig(pydantic.BaseModel):
    module: str
    project: str
    entity: str
    results_dir: str
    trainconfig: Any


def load_logger(config: LogConfig):
    """
    Loads the wandb logger.
    """
    logger = WandbLogger(
        project=config.project,
        entity=config.entity,
        save_dir=config.results_dir,
        name=config.results_dir,
    )
    return logger
