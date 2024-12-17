from datetime import datetime
import json
from types import SimpleNamespace
from typing import List, Optional

from pydantic import BaseModel
from dataclasses import dataclass, asdict

import yaml


class SubUser(BaseModel):
    prop: int


class User(BaseModel):
    id: int
    lr: float
    subuser: SubUser
    name: str = "John Doe"
    signup_ts: Optional[datetime] = None
    friends: List[int] = []


# yaml_str = """
# bill-to: &id-001
#     given  : Chris
#     family : Dumars
# ship-to:
#     order: The order
#     address: *id-001
# """

yaml_str = """
lr: .1
subuser: 
    prop: 10
family : Dumars
ship-to: 
order: The order
address: *id-001
"""


# data = yaml.safe_load(yaml_str)

data_obj = SimpleNamespace(x=10, config=SimpleNamespace(lr=0.1))


# print(80 * "#")
# print("###", "Configuration")
# print(80 * "#")
# print(yaml.safe_dump(data))
# print(80 * "#")

print(data_obj.__dict__)
