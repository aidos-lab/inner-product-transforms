from datetime import datetime
import json
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


yaml_str = """
bill-to: &id-001
    given  : Chris
    family : Dumars
ship-to: 
    order: The order
    address: *id-001
"""

data = yaml.safe_load(yaml_str)
print(data)


# # Parse your YAML into a dictionary, then validate against your model.
# external_data = {
#     "id": "123",
#     "lr": "1e-4",
#     "name": "Hello",
#     "signup_ts": "2019-06-01 12:22",
#     "friends": [1, 2, "3"],
# }
# # # user = User(**external_data)
# user2 = User(**data)
# # # print(user)
# # print(json.dumps(user2))
# print()
