from typing import Tuple, Union

import torch
import torchvision
from torch.utils.flop_counter import FlopCounterMode

from shapesynthesis.loaders import load_config
from shapesynthesis.models.encoder_new import Model, ModelConfig


def get_flops(model, inp: Union[torch.Tensor, Tuple], with_backward=False):

    istrain = model.training
    model.eval()

    inp = inp if isinstance(inp, torch.Tensor) else torch.randn(inp)

    flop_counter = FlopCounterMode(display=False, depth=None)
    with flop_counter:
        if with_backward:
            model(inp).sum().backward()
        else:
            model(inp)
    total_flops = flop_counter.get_total_flops()
    if istrain:
        model.train()
    return total_flops


# ##################################################################
# ### Encoder
# ##################################################################
encoder_config, _ = load_config("configs/encoder_airplane.yaml")
encoder_model = Model(encoder_config.modelconfig)

print(get_flops(encoder_model, (1, 128, 128)))

from torchvision.models import resnet18

model = resnet18()
print(get_flops(model, (1, 3, 224, 224)))
