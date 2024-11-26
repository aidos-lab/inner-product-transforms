import importlib
from types import SimpleNamespace
import yaml
import json

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers import TensorBoardLogger


def load_datamodule(config):
    module = importlib.import_module(config.module)
    model_class = getattr(module, "DataModule")
    return model_class(config)


def load_model(config, model_path=None):
    module = importlib.import_module(config.module)
    model_class = getattr(module, "BaseModel")

    if model_path:
        model = model_class.load_from_checkpoint(model_path)
    else:
        model = model_class(config)
    return model


def load_object(dct):
    return SimpleNamespace(**dct)


def load_config(path):
    with open(path, encoding="utf-8") as stream:
        run_dict = yaml.safe_load(stream)
        config = json.loads(json.dumps(run_dict), object_hook=load_object)
    return config


def get_wandb_logger(config):
    """
    Loads the wandb logger.
    """
    wandb_logger = WandbLogger(
        project=config.project, entity=config.entity, save_dir=config.save_dir
    )

    return wandb_logger


def load_logger(config, logger_type="tensorboard"):
    """
    Loads the wandb logger.
    """
    if logger_type == "wandb":
        logger = WandbLogger(
            project=config.project,
            entity=config.entity,
            save_dir=config.save_dir,
        )
    elif logger_type == "tensorboard":
        logger = TensorBoardLogger(
            "my_logs", name=f"{config.trainer.experimentname}"
        )
    return logger
