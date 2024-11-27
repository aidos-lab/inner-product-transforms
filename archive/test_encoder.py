import json
import argparse
from pprint import pprint

import torch
from metrics.evaluation import EMD_CD
from model_wrapper import ModelWrapperEncoder

from loaders import load_config, load_datamodule, load_model


@torch.no_grad()
def evaluate_recon(model, dm):
    all_sample = []
    all_ref = []
    for batch in dm.test_loader():

        out_pc = model.reconstruct(batch.x.view(-1, 3))

        out_pc = out_pc * batch.std + batch.mean
        te_pc = batch.x.view(-1, 3) * batch.std + batch.mean

        all_sample.append(out_pc)
        all_ref.append(te_pc)

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)

    results = EMD_CD(
        sample_pcs,
        ref_pcs,
        batch_size=16,
        reduced=True,
        accelerated_cd=True,
    )
    results = {
        k: (v.cpu().detach() if not isinstance(v, float) else v)
        for k, v in results.items()
    }
    pprint(results)

    return results


def main(model, dm):
    """
    Perform the reconstruction experiment 10 times
    """
    res = [evaluate_recon(model, dm) for _ in range(10)]
    res_cd = torch.stack([r["MMD-CD"] for r in res])
    res_emd = torch.stack([r["MMD-EMD"] for r in res])
    res_cd_mean = res_cd.mean()
    res_cd_std = res_cd.std()

    res_emd_mean = res_emd.mean()
    res_emd_std = res_emd.std()

    print("===========RESULTS=============")
    print("MMD-CD-Mean", res_cd_mean.item())
    print("MMD-CD-STD", res_cd_std.item())
    print("MMD-EMD-Mean", res_emd_mean.item())
    print("MMD-EMD-STD", res_emd_std.item())
    print("===============================")
    with 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder",
        type=str,
        default=None,
        required=True,
        help="Input encoder configuration",
    )
    parser.add_argument(
        "--vae", type=str, default=None, help="Input vae configuration"
    )

    args = parser.parse_args()

    encoder_config = load_config(args.encoder)
    encoder_model = load_model(encoder_config)

    dm = load_datamodule(encoder_config.data)

    # Ensure that the data section in vae config is the same as in the encoder
    # config, no checks are performed.

    if args.vae:
        vae_config = load_config(args.vae)
        vae_model = load_model(vae_config)
    else:
        vae_model = None

    model = ModelWrapper(encoder_config, vae_config)

    main(
        model,
    )
