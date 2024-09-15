from types import SimpleNamespace
import yaml
import json


def load_object(dct):
    return SimpleNamespace(**dct)


def load_config(path):
    with open(path, encoding="utf-8") as stream:
        run_dict = yaml.safe_load(stream)
        config = json.loads(json.dumps(run_dict), object_hook=load_object)
    return config
