from pathlib import Path
import argparse
import torch

from merge_attend_stable_diffusion_region_based import MergeAttendStableDiffusionRegionBased
from sampling_utils import seed_everything, preprocess_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bg_prompt', type=str, default="A sandy flat beach", help='Prompt for the background.')
    parser.add_argument('--bg_negative', type=str, default="artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image",help='Negative prompt for the background.')
    parser.add_argument('--mask_paths', type=str, nargs='+', default=["./masks/rocks.png", "./masks/bonfire.png"], help='List of paths to mask images for each foreground object.')
    parser.add_argument('--fg_prompts', type=str, nargs='+', default=["Rocks on the beach", "A bonfire with few logs"], help='List of foreground prompts, one for each mask.')
    # Default for fg_negative, will be expanded if only one is provided and multiple fg_prompts exist
    parser.add_argument('--fg_negative', type=str, nargs='+', default=["artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image"], 
                        help='List of negative prompts for foregrounds. Can be a single string repeated.')
    
    parser.add_argument('--model_key', type=str, default='stabilityai/stable-diffusion-2-base')
    parser.add_argument('--H', type=int, default=512)
    parser.add_argument('--W', type=int, default=3072)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--out_name', type=str, default='mad_sd_region_based')
    parser.add_argument('--dtype', type=str, default='fp32')
    parser.add_argument('--mad_threshold', type=int, default=25)
    parser.add_argument('--mad_blocks', type=str, default='all')
    parser.add_argument('--stride', type=int, default=16, help='window stride for MultiDiffusion')
    parser.add_argument('--bootstrapping', type=int, default=20)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    if args.seed is not None:
        seed_everything(args.seed)

    if len(args.fg_negative) == 1 and len(args.fg_prompts) > 1:
        # If only one negative prompt is provided, repeat it for each foreground prompt
        args.fg_negative = args.fg_negative * len(args.fg_prompts)

    # Load Stable Diffusion model
    model = MergeAttendStableDiffusionRegionBased(args.model_key, device, args.dtype)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    out_file = Path(args.out_name).stem

    # Preprocess masks and prompts
    fg_masks = torch.cat([preprocess_mask(mask_path, args.H // 8, args.W // 8, device) for mask_path in args.mask_paths])
    bg_mask = 1 - torch.sum(fg_masks, dim=0, keepdim=True)
    bg_mask[bg_mask < 0] = 0
    masks = torch.cat([bg_mask, fg_masks])
    prompts = [args.bg_prompt] + args.fg_prompts
    neg_prompts = [args.bg_negative] + args.fg_negative

    print(f"{device} - Background Prompt: {args.bg_prompt}")
    try:
        for sample_idx in range(args.num_samples):
            save_path = Path(save_dir) / args.bg_prompt.replace(' ', '_') / f"{out_file}_{sample_idx:04d}.png"
            save_path.parent.mkdir(exist_ok=True, parents=True)

            # Generate images
            img = model.sample(
                masks=masks,
                prompts=prompts,
                negative_prompts=neg_prompts,
                bootstrapping=args.bootstrapping,
                height=args.H,
                width=args.W,
                num_inference_steps=args.steps,
                stride=args.stride,
                mad_blocks=args.mad_blocks,
                mad_threshold=args.mad_threshold
            )
            img.save(save_path)
            torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print(f"Interrupted! Saving results...")
    print(f"Done!")


if __name__ == "__main__":
    main()
