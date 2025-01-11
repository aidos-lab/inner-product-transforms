"""
Experiment to build the following table.
For each element in the table there is a config file.

|         | ECT-64 |     |       | ECT-128 |       |     | ECT256 |       |     |
| ------- | ------ | --- | ----- | ------- | ----- | --- | ------ | ----- | --- |
| Dataset | Air    | Car | Chair | Air     | Chair | Car | Air    | Chair | Car |
| Scale   |        |     |       |         |       |     |        |       |     |
| 25%     |        |     |       |         |       |     |        |       |     |
| 50%     |        |     |       |         |       |     |        |       |     |
| 75%     |        |     |       |         |       |     |        |       |     |
| 100%    |        |     |       |         |       |     |        |       |     |

"""

import argparse
import json
from pprint import pprint
from typing import Any, Callable, Tuple

import numpy as np
import torch
from datasets.shapenetcore import DataModuleConfig as DataConfig
from layers.directions import generate_uniform_directions
from layers.ect import EctConfig, compute_ect_point_cloud
from loaders import load_config, load_datamodule
from metrics.evaluation import EMD_CD
from pydantic import BaseModel
from renderers.pointcloud import render_point_cloud

torch.set_float32_matmul_precision("medium")
DTYPE = torch.float32


class RenderConfig(BaseModel):
    num_pts: int
    num_epochs: int
    dataset_name: str
    experiment_name: str
    ectconfig: EctConfig


def render_point_clouds(
    dm,
    dataconfig: DataConfig,
    renderconfig: RenderConfig,
    render_point_cloud_compiled: Callable,
    fast_run: bool = False,
) -> Tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    loader = dm.test_dataloader()
    total = len(dm.test_ds)

    # Set up the ECT configuration.
    v = (
        generate_uniform_directions(
            renderconfig.ectconfig.num_thetas,
            renderconfig.ectconfig.ambient_dimension,
            renderconfig.ectconfig.seed,
        )
        .type(DTYPE)
        .cuda()
    )

    x_rendered_pcs, x_gt_pcs = [], []
    for batch_idx, test_batch in enumerate(loader):
        if fast_run and batch_idx == 1:
            break

        test_batch.cuda()
        x_gt = (
            test_batch.x.view(
                -1,
                dataconfig.num_pts,
                dataconfig.ectconfig.ambient_dimension,
            )
            .type(DTYPE)
            .cuda()
        )
        pc_shape = x_gt.shape
        print(f"Processing idx {batch_idx} out of {total // dataconfig.batch_size}")
        x_init = (renderconfig.ectconfig.r / 2) * (
            torch.rand(
                size=(
                    len(x_gt),
                    renderconfig.num_pts,
                    renderconfig.ectconfig.ambient_dimension,
                ),
                dtype=DTYPE,
            )
            - 0.5
        ).cuda()
        ect_gt = compute_ect_point_cloud(
            x_gt,
            v,
            radius=renderconfig.ectconfig.r,
            resolution=renderconfig.ectconfig.resolution,
            scale=renderconfig.ectconfig.scale,
        )

        x_rendered = render_point_cloud_compiled(
            x_init,
            ect_gt,
            v,
            renderconfig.num_epochs,
            scale=renderconfig.ectconfig.scale,
            radius=renderconfig.ectconfig.r,
            resolution=renderconfig.ectconfig.resolution,
        ).cpu()

        # m, s = data["mean"].float(), data["std"].float()
        if hasattr(test_batch, "mean") and hasattr(test_batch, "std"):
            m = torch.tensor(np.stack(test_batch.mean))
            s = torch.tensor(np.stack(test_batch.std))
        else:
            m = torch.zeros(size=(1, 1, pc_shape[-1]))
            s = torch.ones(size=(1, 1, 1))

        x_gt = x_gt.cpu() * s + m
        x_rendered = x_rendered * s + m

        x_rendered_pcs.append(x_rendered)
        x_gt_pcs.append(x_gt)

    x_rendered_pcs = torch.cat(x_rendered_pcs)
    x_gt_pcs = torch.cat(x_gt_pcs)

    results = EMD_CD(
        x_rendered_pcs,
        x_gt_pcs,
        batch_size=8,
        reduced=True,
        accelerated_cd=True,
    )
    results = {
        ("%s" % k): (v if isinstance(v, float) else v.item())
        for k, v in results.items()
    }

    return results, x_rendered_pcs, x_gt_pcs


def main():
    """
    Builds the argparser and parses them.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Input configuration")
    parser.add_argument("--num_reruns", type=int, default=1, help="Number of reruns.")
    parser.add_argument(
        "--fast-run",
        default=False,
        action="store_true",
        help="Run only a few batches",
    )
    args = parser.parse_args()

    config, _ = load_config(args.config)

    dm = load_datamodule(config.data)
    render_point_cloud_compiled = torch.compile(render_point_cloud)
    results = []
    for _ in range(args.num_reruns):
        result, x_rendered_pcs, x_gt_pcs = render_point_clouds(
            dm,
            config.data,
            config.render,
            render_point_cloud_compiled,
            fast_run=args.fast_run,
        )

        # Add extra information to the result.
        result["model"] = (
            f"render_{config.render.experiment_name}_{config.render.dataset_name}"
        )
        results.append(result)

    pprint(results)

    with open(
        f"./results/rendered_ect/{config.render.dataset_name}/{config.render.experiment_name}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results, f)

    torch.save(
        x_rendered_pcs,  # type: ignore
        f"./results/rendered_ect/{config.render.dataset_name}/{config.render.experiment_name}_reconstructions.pt",
    )
    torch.save(
        x_gt_pcs,  # type: ignore
        f"./results/rendered_ect/{config.render.dataset_name}/references.pt",
    )


if __name__ == "__main__":
    main()
