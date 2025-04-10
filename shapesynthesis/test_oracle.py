import json

import torch
import torch.utils.data
from datasets.shapenet_data_pc import ShapeNet15kPointClouds
from metrics.evaluation import EMD_CD

# Settings
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_datasets(category, dataroot, npoints=2048):
    tr_dataset = ShapeNet15kPointClouds(
        root_dir=dataroot,
        categories=category,
        split="train",
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.0,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        random_subsample=True,
    )
    te_dataset = ShapeNet15kPointClouds(
        root_dir=dataroot,
        categories=category,
        split="val",
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.0,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        all_points_mean=tr_dataset.all_points_mean,
        all_points_std=tr_dataset.all_points_std,
        # random_subsample=True,
    )
    return te_dataset


# ############################################################################ #
#                                    Script                                    #
# ############################################################################ #

# dm = load_datamodule(encoder_config.data)
final_results = []
for cate in ["airplane", "chair", "car"]:
    for _ in range(5):
        te_ds = get_datasets(
            [cate],
            "./data/shapenet/raw/ShapeNetCore.v2.PC15k",
        )
        loader = torch.utils.data.DataLoader(te_ds, batch_size=8)

        all_references = []
        all_reconstructions = []

        for batch in loader:

            pc_test = batch["test_points"].cuda()
            pc_train = batch["train_points"].cuda()

            s = batch["std"].cuda()
            m = batch["mean"].cuda()

            pc_test = pc_test * s + m
            pc_train = pc_train * s + m

            all_references.append(pc_test)
            all_reconstructions.append(pc_train)

        sample_pcs = torch.cat(all_reconstructions, dim=0)
        ref_pcs = torch.cat(all_references, dim=0)

        results = EMD_CD(
            sample_pcs,
            ref_pcs,
            batch_size=8,
            reduced=True,
            accelerated_cd=True,
        )

        results = {
            ("%s" % k): (v if isinstance(v, float) else v.item())
            for k, v in results.items()
        }
        results["model"] = f"oracle_{cate}"
        results["normalized"] = False

        final_results.append(results)

with open("./results/oracle/results.json", "w", encoding="utf-8") as f:
    json.dump(final_results, f)
