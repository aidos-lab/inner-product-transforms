import json
from dataclasses import asdict, dataclass
from types import SimpleNamespace

from shapesynthesis.loaders import load_config

config, _ = load_config("./configs/encoder_new_airplane.yaml")
print(config.modelconfig)
