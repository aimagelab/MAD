# MAD: Semantically Coherent Montages by Merging and Splitting Diffusion Paths
Official PyTorch implementation for "Semantically Coherent Montages by Merging and Splitting Diffusion Paths", presenting the Merge-Attend-Diffuse operator

The code is tested on Python 3.11.5, CUDA 12.1, and PyTorch 2.1.1

## Installation
This is the list of python packages that we need to run inference 
```console
conda create --name mad python=3.11
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install diffusers["torch"] transformers
```


## Inference with Stable Diffusion
Basic code to run inference with the default parameters
```
python sample_panorama_stable_diffusion.py
```

## Inference with LCM
Basic code to run inference with the default parameters
```
python sample_panorama_lcm.py
```

## Acknowledgements
Our code is heavily based on the [implementation](https://github.com/omerbt/MultiDiffusion) of [MultiDiffusion](https://multidiffusion.github.io/)
