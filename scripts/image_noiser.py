import argparse
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    regressor_and_diffusion_defaults,
    create_regressor_and_diffusion,
)
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import numpy as np

def main():
    args = create_argparser().parse_args()

    batch = np.load(args.topo_dir)
    t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev())
    batch = diffusion.q_sample(batch, t)
    
    np.save(args.out_dir + "/noisy_images.npy", batch)
    np.save(args.out_dir + "/timesteps.npy", t)

def create_argparser():
    defaults = dict(
        topo_dir="",
        out_dir="",
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
