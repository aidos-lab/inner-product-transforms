import json

import yaml


def print_config(config):
    print(yaml.dump(json.loads(json.dumps(config, default=lambda s: vars(s)))))
