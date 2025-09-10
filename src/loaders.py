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


def load_datamodule(config, dev: bool = False):
    module = importlib.import_module(config.module)
    train_dl, val_dl, test_dl, m, s = module.get_all_dataloaders(config, dev=dev)
    return SimpleNamespace(
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        test_dataloader=test_dl,
        m=m,
        s=s,
    )


def load_model(config, model_path=None):
    module = importlib.import_module(config.module)
    model_class = getattr(module, "BaseLightningModel")

    if model_path:
        model = model_class.load_from_checkpoint(model_path)
    else:
        config_dict = json.loads(json.dumps(config, default=lambda s: vars(s)))
    model = model_class(config)
    return model


def load_transform(config):
    module = importlib.import_module(config.module)
    transform_class = getattr(module, "Transform")
    transform = transform_class(config)
    return transform


# @timeit_decorator
def load_object(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**obj)
    else:
        return obj


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
    Loads the logger.
    """
    module = importlib.import_module(config.module)
    return module.load_logger(config)
