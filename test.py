import argparse
import json
from pprint import pprint
import torch

from model_wrapper import ModelWrapper
from loaders import load_config, load_datamodule, load_model
from metrics.evaluation import EMD_CD

# Settings
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def evaluate_reconstruction(model, dm):
    all_sample = []
    all_ref = []
    for i, batch in enumerate(dm.test_dataloader()):
        out_pc = model.reconstruct(batch.to(DEVICE))
        pc_shape = batch[0].x.shape

        # m, s = data["mean"].float(), data["std"].float()
        # m = m.cuda() if args.gpu is None else m.cuda(args.gpu)
        # s = s.cuda() if args.gpu is None else s.cuda(args.gpu)
        # out_pc = out_pc * s + m
        # te_pc = te_pc * s + m

        # if args.normalize:
        #     te_pc, means, norms = normalize(te_pc.clone())
        #     out_pc -= means
        #     out_pc /= norms

        # if i == 0:
        #     print("Train PC", tr_pc[0].view(-1, 3).norm(dim=-1).max())
        #     print("Test PC", te_pc[0].view(-1, 3).norm(dim=-1).max())
        #     print("OUT PC", out_pc[10].view(-1, 3).norm(dim=-1).max())

        all_sample.append(out_pc)
        all_ref.append(batch.x.view(-1, pc_shape[0], pc_shape[1]))

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)

    results = EMD_CD(
        sample_pcs,
        ref_pcs,
        args.batch_size,
        reduced=True,
        accelerated_cd=True,
    )

    results = {
        ("%s" % k): (v if isinstance(v, float) else v.item())
        for k, v in results.items()
    }

    return results, sample_pcs, ref_pcs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder_config",
        required=True,
        type=str,
        help="Encoder configuration",
    )
    parser.add_argument(
        "--vae_config",
        required=False,
        default=None,
        type=str,
        help="VAE Configuration (Optional)",
    )
    parser.add_argument(
        "--num_reruns",
        default=1,
        type=int,
        help="Number of reruns for the standard deviation.",
    )
    parser.add_argument(
        "--generative",
        default=False,
        action="store_true",
        help="Evaluation generative performance.",
    )
    parser.add_argument(
        "--normalize",
        default=False,
        action="store_true",
        help="Normalize data before passing it to the model.",
    )
    args = parser.parse_args()

    encoder_config = load_config(args.encoder_config)
    encoder_model = load_model(
        encoder_config.modelconfig,
        f"./{encoder_config.trainer.save_dir}/{encoder_config.trainer.model_name}",
    ).to(DEVICE)

    dm = load_datamodule(encoder_config.data)

    # Load the datamodule
    # NOTE: Loads the datamodule from the encoder and does not check for
    # equality of the VAE data configs.

    if args.vae_config:
        vae_config = load_config(args.encoder_config)
        vae_model = load_model(vae_config.model_config)
    else:
        vae_model = None

    model = ModelWrapper(encoder_model, vae_model)

    # for i, batch in enumerate(dm.test_dataloader()):
    #     out_pc = model.reconstruct(batch.to(DEVICE))
    #     break

    results = []
    for _ in range(args.num_reruns):
        # Evaluate reconstruction
        result, sample_pc, ref_pc = evaluate_reconstruction(model, dm)
        result["normalized"] = args.normalize

        results.append(result)

    # if args.normalize:
    #     suffix = "_normalized"
    # else:
    #     suffix = ""

    pprint(results)
    # with open(
    #     f"./results/{args.model}/{args.cates[0]}{suffix}.json",
    #     "w",
    #     encoding="utf-8",
    # ) as f:
    #     json.dump(results, f)
    # torch.save(
    #     sample_pc,
    #     f"./results/{args.model}/samples_{args.cates[0]}{suffix}.pt",
    # )
    # torch.save(
    #     ref_pc,
    #     f"./results/{args.model}/ref_{args.cates[0]}{suffix}.pt",
    # )


# #
# def main(encoder_config, vae_config=None):
#     results = []
#     if args.evaluate_recon:
#         for _ in range(args.num_reruns):
#             # Evaluate reconstruction
#             result, sample_pc, ref_pc = evaluate_recon(model, args)
#             result["model"] = args.model
#             result["cate"] = args.cates[0]
#             result["normalized"] = args.normalize

#             results.append(result)

#         suffix = ""
#         if args.normalize:
#             suffix = "_normalized"

#         pprint(results)
#         with open(
#             f"./results/{args.model}/{args.cates[0]}{suffix}.json",
#             "w",
#             encoding="utf-8",
#         ) as f:
#             json.dump(results, f)
#         torch.save(
#             sample_pc,
#             f"./results/{args.model}/samples_{args.cates[0]}{suffix}.pt",
#         )
#         torch.save(
#             ref_pc,
#             f"./results/{args.model}/ref_{args.cates[0]}{suffix}.pt",
#         )
