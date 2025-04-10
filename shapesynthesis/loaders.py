import functools
import importlib
import json
import timeit
from types import SimpleNamespace

import yaml
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger


def timeit_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        timer = timeit.Timer(lambda: func(*args, **kwargs))
        execution_time = timer.timeit(number=1)
        print(f"Function {func.__name__!r} executed in {execution_time:.4f} seconds")
        return func(*args, **kwargs)

    return wrapper


# @timeit_decorator
def load_datamodule(config, debug: bool = False):
    module = importlib.import_module(config.module)
    model_class = getattr(module, "DataModule")
    return model_class(config, debug)


# @timeit_decorator
def load_model(config, model_path=None):
    module = importlib.import_module(config.module)
    model_class = getattr(module, "BaseLightningModel")
    config_class = getattr(module, "ModelConfig")

    if model_path:
        model = model_class.load_from_checkpoint(model_path)
    else:
        print(config.__dict__)
        config_dict = json.loads(json.dumps(config, default=lambda s: vars(s)))
        config = config_class(**config_dict)
        model = model_class(config)
    return model


# @timeit_decorator
def load_object(dct):
    return SimpleNamespace(**dct)


# @timeit_decorator
def load_config(path):
    """
    Loads the configuration yaml and parses it into an object with dot access.
    """
    with open(path, encoding="utf-8") as stream:
        # Load dict
        config_dict = yaml.safe_load(stream)

        # Convert to namespace (access via config.data etc)
        config = json.loads(json.dumps(config_dict), object_hook=load_object)
    return config, config_dict


def validate_configuration(run_config_dict: dict):
    """
    Loads the pydantic configuration object and checks if it is valid. This
    ensures we can test all configurations for missing keys etc, before running
    the experiments.
    """

    # Test the model config
    module = importlib.import_module(run_config_dict["modelconfig"]["module"])
    model_class = getattr(module, "BaseLightningModel")
    config_class = getattr(module, "ModelConfig")
    config_class(**run_config_dict["modelconfig"])

    # Test the dataset
    module = importlib.import_module(run_config_dict["data"]["module"])
    model_class = getattr(module, "DataModule")
    config_class = getattr(module, "DataModuleConfig")
    config_class(**run_config_dict["data"])

    assert "logger" in run_config_dict["loggers"]
    assert "tags" in run_config_dict["loggers"]


# @timeit_decorator
def get_wandb_logger(config):
    """
    Loads the wandb logger.
    """
    wandb_logger = WandbLogger(
        project=config.project, entity=config.entity, save_dir=config.save_dir
    )

    return wandb_logger


def load_logger(config):
    """
    Loads the wandb logger.
    """
    if config.logger == "wandb":
        logger = WandbLogger(
            project=config.project,
            entity=config.entity,
            save_dir=config.save_dir,
            name=config.experiment_name,
            tags=config.tags,
        )
    elif config.logger == "tensorboard":
        logger = TensorBoardLogger("my_logs", name=f"{config.experiment_name}")
    return logger
