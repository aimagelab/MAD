import os
from os.path import join
import torch
import argparse
import json
import logging
import random
import numpy as np

from itertools import product
import torch
from accelerate import PartialState
from global_attention_region import GlobalAttentionRegion
from sampling_utils import preprocess_mask, seed_everything

from regions import regions_settings

torch.backends.cuda.matmul.allow_tf32 = True

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--mask_paths', type=list)  # important: it is necessary that SD output high-quality images for the bg/fg prompts.
    # parser.add_argument('--bg_prompt', type=str)
    # parser.add_argument('--bg_negative', type=str)  # 'artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image'
    # parser.add_argument('--fg_prompts', type=list)
    # parser.add_argument('--fg_negative', type=list)  # 'artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image'

    parser.add_argument('--H', type=int, default=512)
    parser.add_argument('--W', type=int, default=3072)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument("--force_resampling", action="store_true", help="Regenerate samples even if they exist")
    parser.add_argument("--activate_tqdm", action="store_true")
    parser.add_argument('--dtype', type=str, default='fp32')

    parser.add_argument('--global_self_attention_thres', type=int, required=True)
    parser.add_argument('--where_global_self_attention', type=str, required=True)
    parser.add_argument('--ca_type', type=str, required=True)
    parser.add_argument('--sa_type', type=str, required=True)
    parser.add_argument('--sa_base_type', type=str, default='batch')
    parser.add_argument("--activate_region_based_guidance", action="store_true")

    parser.add_argument('--stride', type=int, default=16, help="window stride for MultiDiffusion")
    # bootstrapping encourages high fidelity to tight masks, the value can be lowered is most cases
    parser.add_argument('--bootstrapping', type=int, default=20)
    parser.add_argument('--seed', type=int, default=2023)

    args = parser.parse_args()
    id = 2

    args.mask_paths = regions_settings['first_map'][0]['mask_paths']
    args.fg_prompts = regions_settings['first_map'][0]['fg_prompts']
    args.fg_negative = regions_settings['first_map'][0]['fg_negative']
    args.bg_prompt = regions_settings['first_map'][0]['bg_prompt']
    args.bg_negative = regions_settings['first_map'][0]['bg_negative']

    args.where_global_self_attention = [str(v) for v in args.where_global_self_attention.split(",")]

    distributed_state = PartialState()
    args.device = distributed_state.device
    logger.info(f"Using device {args.device}")

    seed_everything(args.seed)

    model = GlobalAttentionRegion(args.device, dtype=args.dtype)
    os.makedirs(args.save_dir, exist_ok=True)

    fg_masks = torch.cat([preprocess_mask(mask_path, args.H // 8, args.W // 8, args.device) for mask_path in args.mask_paths])
    bg_mask = 1 - torch.sum(fg_masks, dim=0, keepdim=True)
    bg_mask[bg_mask < 0] = 0
    masks = torch.cat([bg_mask, fg_masks])
    prompts = [args.bg_prompt] + args.fg_prompts
    neg_prompts = [args.bg_negative] + args.fg_negative

    sampling_params = {
        "global_self_attention_thres": args.global_self_attention_thres,
        "where_global_self_attention": args.where_global_self_attention,
        'ca_type': args.ca_type,
        'sa_type': args.sa_type,
        'sa_base_type': args.sa_base_type,
        'prompts': prompts,
        'negative_prompts': neg_prompts,
        'bootstrapping': args.bootstrapping,
        'activate_region_based_guidance': args.activate_region_based_guidance,
    }


    logging.info(f"{args.device}")

    save_params = {
        "seed": args.seed
    }

    for sample_idx in range(args.num_samples):
        save_path = join(args.save_dir, f"{sample_idx:04d}.png")

        if os.path.exists(save_path):
            continue

        # todo save better the params

        # Generate images
        img = model.generate(
            masks=masks,
            height=args.H,
            width=args.W,
            num_inference_steps=args.steps,
            stride=args.stride,
            disable_tqdm=not args.activate_tqdm,
            **sampling_params
        )

        img.save(save_path)
        logger.info(f"{args.device} - Saved image to {save_path}")

    logger.info(f"Done!")



if __name__ == "__main__":
    main()
