"""
Like image_sample.py, but use a noisy image regressor to guide the sampling
process towards more realistic images.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from guided_diffusion.cons_input_datasets import load_data
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    regressor_defaults,
    create_model_and_diffusion,
    create_regressor,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading regressor...")
    regressor = create_regressor(**args_to_dict(args, regressor_defaults().keys()))
    regressor.load_state_dict(
        dist_util.load_state_dict(args.regressor_path, map_location="cpu")
    )
    regressor.to(dist_util.dev())
    if args.regressor_use_fp16:
        regressor.convert_to_fp16()
    regressor.eval()

    data = load_data(
        data_dir=args.constraints_path, #CAUTION : REQUIRES THE BATCH SIZE TO BE 1
    )

    def cond_fn(x, t):
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = regressor(x_in, t)
            #log_probs = F.log_softmax(logits, dim=-1)
            #selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(logits.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t):
        return model(x, t)

    logger.log("sampling...")
    all_images = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        input_cons = next(data).cuda()
        sample = sample_fn(
            model_fn,
            (args.batch_size, 1, args.image_size, args.image_size),
            input_cons,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
        )
        #sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = (sample * 255).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=20,
        batch_size=1,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(regressor_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
