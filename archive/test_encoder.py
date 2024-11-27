"""Evaluation"""

from copy import Error
import os
from types import SimpleNamespace
import json
import argparse
from pprint import pprint

import torch
import numpy as np

from datasets_pointflow import get_datasets, synsetid_to_cate
from metrics.evaluation_metrics import EMD_CD
from metrics.evaluation_metrics import jsd_between_point_cloud_sets as JSD
from metrics.evaluation_metrics import compute_all_metrics

from model_wrapper import ModelWrapperEncoder, ModelWrapperVAE
from load_models import load_encoder


def load_object(dct):
    return SimpleNamespace(**dct)


def get_test_loader(pointflow_config):
    _, te_dataset = get_datasets(pointflow_config)
    if (
        pointflow_config.resume_dataset_mean is not None
        and args.resume_dataset_std is not None
    ):
        mean = np.load(pointflow_config.resume_dataset_mean)
        std = np.load(pointflow_config.resume_dataset_std)
        te_dataset.renormalize(mean, std)
    loader = torch.utils.data.DataLoader(
        dataset=te_dataset,
        batch_size=pointflow_config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    return loader


def evaluate_recon(model, pointflow_config):
    # TODO: make this memory efficient
    if "all" in pointflow_config.cates:
        cates = list(synsetid_to_cate.values())
    else:
        cates = pointflow_config.cates
    all_results = {}
    cate_to_len = {}
    save_dir = os.path.dirname(pointflow_config.resume_checkpoint)
    for cate in cates:
        pointflow_config.cates = [cate]
        loader = get_test_loader(pointflow_config)

        all_sample = []
        all_ref = []
        for data in loader:
            _, tr_pc, te_pc = data["idx"], data["train_points"], data["test_points"]

            te_pc = (
                te_pc.cuda() if pointflow_config.gpu is None else te_pc.cuda(args.gpu)
            )
            tr_pc = (
                tr_pc.cuda() if pointflow_config.gpu is None else tr_pc.cuda(args.gpu)
            )
            _, N = te_pc.size(0), te_pc.size(1)
            out_pc = model.reconstruct(te_pc, num_points=N)

            m, s = data["mean"].float(), data["std"].float()
            m = m.cuda() if pointflow_config.gpu is None else m.cuda(args.gpu)
            s = s.cuda() if pointflow_config.gpu is None else s.cuda(args.gpu)
            out_pc = out_pc * s + m
            te_pc = te_pc * s + m

            all_sample.append(out_pc)
            all_ref.append(te_pc)

        sample_pcs = torch.cat(all_sample, dim=0)
        ref_pcs = torch.cat(all_ref, dim=0)

        cate_to_len[cate] = int(sample_pcs.size(0))
        print(
            "Cate=%s Total Sample size:%s Ref size: %s"
            % (cate, sample_pcs.size(), ref_pcs.size())
        )

        # Save it
        np.save(
            os.path.join(save_dir, "%s_out_smp.npy" % cate),
            sample_pcs.cpu().detach().numpy(),
        )
        np.save(
            os.path.join(save_dir, "%s_out_ref.npy" % cate),
            ref_pcs.cpu().detach().numpy(),
        )

        results = EMD_CD(
            sample_pcs,
            ref_pcs,
            pointflow_config.batch_size,
            reduced=True,
            accelerated_cd=True,
        )
        results = {
            k: (v.cpu().detach() if not isinstance(v, float) else v)
            for k, v in results.items()
        }
        pprint(results)
        all_results[cate] = results

    # Save final results
    print("=" * 80)
    print("All category results:")
    print("=" * 80)
    pprint(all_results)
    save_path = os.path.join(save_dir, "percate_results.npy")
    np.save(save_path, all_results)

    return all_results


def evaluate_gen(model, pointflow_config):
    loader = get_test_loader(pointflow_config)
    all_sample = []
    all_ref = []
    for data in loader:
        idx_b, te_pc = data["idx"], data["test_points"]
        te_pc = te_pc.cuda() if pointflow_config.gpu is None else te_pc.cuda(args.gpu)
        B, N = te_pc.size(0), te_pc.size(1)
        _, out_pc = model.sample(B, N)

        # denormalize
        m, s = data["mean"].float(), data["std"].float()
        m = m.cuda() if pointflow_config.gpu is None else m.cuda(args.gpu)
        s = s.cuda() if pointflow_config.gpu is None else s.cuda(args.gpu)
        out_pc = out_pc * s + m
        te_pc = te_pc * s + m

        all_sample.append(out_pc)
        all_ref.append(te_pc)

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)
    print(
        "Generation sample size:%s reference size: %s"
        % (sample_pcs.size(), ref_pcs.size())
    )

    # Save the generative output
    save_dir = os.path.dirname(pointflow_config.resume_checkpoint)
    np.save(
        os.path.join(save_dir, "model_out_smp.npy"), sample_pcs.cpu().detach().numpy()
    )
    np.save(os.path.join(save_dir, "model_out_ref.npy"), ref_pcs.cpu().detach().numpy())

    # Compute metrics
    results = compute_all_metrics(
        sample_pcs, ref_pcs, pointflow_config.batch_size, accelerated_cd=True
    )
    results = {
        k: (v.cpu().detach().item() if not isinstance(v, float) else v)
        for k, v in results.items()
    }
    pprint(results)

    sample_pcl_npy = sample_pcs.cpu().detach().numpy()
    ref_pcl_npy = ref_pcs.cpu().detach().numpy()
    jsd = JSD(sample_pcl_npy, ref_pcl_npy)
    print("JSD:%s" % jsd)


def main(pointflow_config, args):
    """Main"""


    with torch.no_grad():
        if args.recon and not args.gen:
            encoder_model = load_encoder(args.model)
            model = ModelWrapperEncoder(encoder_model)
            # Evaluate reconstruction
            res = [evaluate_recon(model, pointflow_config) for _ in range(10)]
            res_cd = torch.stack([r[pointflow_config.cates[0]]["MMD-CD"] for r in res])
            res_emd = torch.stack(
                [r[pointflow_config.cates[0]]["MMD-EMD"] for r in res]
            )
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
            torch.save(res, "res_encoder.pt")

        elif args.gen and not args.recon:
            encoder_model = load_encoder(args.model)
            vae_model = load_encoder(args.model)
            model = ModelWrapperVAE(encoder_model)
            evaluate_gen(model, pointflow_config)
        else:
            raise ValueError("Error")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="Input configuration")
    parser.add_argument(
        "--recon", default=False, action="store_true", help="Run a test batch"
    )
    parser.add_argument(
        "--gen", default=False, action="store_true", help="Run a test batch"
    )
    parser.add_argument(
        "--cates",
        type=str,
        nargs="+",
        default=["airplane"],
        help="Categories to be trained (useful only if 'shapenet' is selected)",
    )

    args = parser.parse_args()

    with open("args.json", encoding="utf-8") as f:
        pointflow_cfg = json.load(f, object_hook=load_object)

    pointflow_cfg.resume_checkpoint = "."
    pointflow_cfg.cates = args.cates
    pointflow_cfg.data_dir = "./data/shapenet/raw/ShapeNetCore.v2.PC15k"
    main(pointflow_cfg, args)
