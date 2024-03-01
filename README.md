# MAD: Semantically Coherent Montages by Merging and Splitting Diffusion Paths
Official PyTorch implementation for "Semantically Coherent Montages by Merging and Splitting Diffusion Paths", presenting the Merge-Attend-Diffuse operator

The code is tested on Python 3.11.5, CUDA 12.1, and PyTorch 2.1.1

## Installation
This is the list of python packages that we need to run inference 
```console
conda create --name mad python=3.11
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install diffusers["torch"] transformers einops
```


## Inference with Stable Diffusion
Basic code to run inference with the default parameters
```
python sample_panorama_stable_diffusion.py
```

Some suggestions:
```
python sample_panorama_stable_diffusion.py --prompt "A shelf full of colorful books"

python sample_panorama_stable_diffusion.py --prompt "Tube map of London"

python sample_panorama_stable_diffusion.py --prompt "A whole shepherd pie"
```

## Inference with LCM
Basic code to run inference with the default parameters
```
python sample_panorama_lcm.py
```

Some suggestions:
```
python sample_panorama_lcm.py --prompt "A pride concert full of colorful fireworks"

python sample_panorama_lcm.py --prompt "Top-view of a square pizza"
```

Some suggestions of vertical images:
```
python sample_panorama_lcm.py --prompt "A tower in a colorful sky" --W 512 --H 2048

python sample_panorama_lcm.py --prompt "A view of a river inside a canyon" --W 512 --H 2048
```

## Acknowledgements
Our code is heavily based on the [implementation](https://github.com/omerbt/MultiDiffusion) of [MultiDiffusion](https://multidiffusion.github.io/)
