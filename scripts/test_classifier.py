import argparse
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import numpy as np

from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    regressor_defaults,
    model_and_diffusion_defaults,
    create_regressor_and_diffusion,
    create_model_and_diffusion,
    create_regressor,
)
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to("cpu")#dist_util.dev())

    regressor = create_regressor(**args_to_dict(args, regressor_defaults().keys()))
    regressor.load_state_dict(
        dist_util.load_state_dict(args.regressor_path, map_location="cpu")
    )
    regressor.to("cpu")#dist_util.dev())
    #if args.regressor_use_fp16:
    #    regressor.convert_to_fp16()
    regressor.eval()

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        random_crop=False
    )

    def compute_deflection(data_loader, timestep = 0, prefix="test", print_res=False):
        batch, batch_cons, extra = next(data_loader)
        deflect = extra["d"].to("cpu")#dist_util.dev())
        
        batch = batch.to("cpu")#dist_util.dev())
        batch_cons = batch_cons.to("cpu")#dist_util.dev())
        # Noisy images
        if args.noised:
            t = timestep * th.ones(batch.shape[0], dtype=th.long, device="cpu")#dist_util.dev())
            batch = diffusion.q_sample(batch, t)
        else:
            t = th.zeros(batch.shape[0], dtype=th.long, device="cpu")#dist_util.dev())
        for i, (sub_batch, sub_batch_cons, sub_deflect, sub_t) in enumerate(
            split_microbatches(args.microbatch, batch, batch_cons, deflect, t)
        ):
            full_batch = th.cat((sub_batch, sub_batch_cons), dim=1)
            logits = regressor(full_batch, timesteps=sub_t).reshape(sub_deflect.shape) ##WARNING !!!
            if print_res:
                print(logits, sub_deflect)
            #loss = F.cross_entropy(logits, sub_labels, reduction="none")
            loss = F.mse_loss(logits, sub_deflect)
            #losses = {}
            #losses[f"{prefix}_loss"] = loss.detach()
            R2score = r2_score(sub_deflect.cpu().detach().numpy(), logits.cpu().detach().numpy())
        return loss, R2score

    losses = np.empty(20)
    R2scores = np.empty(20)
    for timestep in range(20):
        print(f"Computing timestep {timestep}")
        loss, R2 = compute_deflection(data, timestep = timestep)
        losses[timestep] = loss
        R2scores[timestep] = R2
    np.save("logdir/losses.npy", losses)
    np.save("logdir/R2scores.npy", R2scores)

def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)


def create_argparser():
    defaults = dict(
        data_dir="",
        noised=True,
        batch_size=4,
        microbatch=-1,
        regressor_path="",
    )
    defaults.update(regressor_defaults())
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()