from pathlib import Path
import argparse
import torch

from merge_attend_stable_diffusion_xl import StableDiffusionXLPipeline
from sampling_utils import seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='A fancy bathroom')
    parser.add_argument('--negative', type=str, default='')
    parser.add_argument('--model_key', type=str, default='stabilityai/stable-diffusion-xl-base-1.0')
    parser.add_argument('--H', type=int, default=1024)
    parser.add_argument('--W', type=int, default=4096)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--out_name', type=str, default='mad_sdxl')
    parser.add_argument('--dtype', type=str, default='fp32')
    parser.add_argument('--mad_threshold', type=int, default=25)
    parser.add_argument('--mad_blocks', type=str, default='all')
    parser.add_argument('--stride', type=int, default=16, help='window stride for MultiDiffusion')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    if args.seed is not None:
        seed_everything(args.seed)

    model = StableDiffusionXLPipeline.from_pretrained(args.model_key, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to(device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    out_file = Path(args.out_name).stem

    print(f"{device} - Prompt: {args.prompt}")
    try:
        for sample_idx in range(args.num_samples):
            save_path = Path(save_dir) / args.prompt.replace(' ', '_') / f"{out_file}_{sample_idx:04d}.png"
            save_path.parent.mkdir(exist_ok=True, parents=True)
    
            # Generate images
            img = model(
                prompt= args.prompt,
                height= args.H,
                width= args.W,
                num_inference_steps= args.steps,
                stride= args.stride,
                mad_blocks= args.mad_blocks,
                mad_threshold= args.mad_threshold
            )[0][0]

            img.save(save_path)
            torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print(f"Interrupted! Saving results...")
    print(f"Done!")


if __name__ == "__main__":
    main()
