import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from tqdm import tqdm

from sampling_utils import *
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, DDPMScheduler
from diffusers import LCMScheduler

from nn_utils import CrossViewsAttnProcessor2_0


class MergeAttendLCM(nn.Module):
    def __init__(self, model_key, device='cuda', dtype='fp32'):
        super().__init__()

        self.device = device
        self.dtype = torch.float16 if dtype == 'fp16' else torch.float32

        print('Loading LCM...')
        # Load pretrained models from HuggingFace
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.device, self.dtype)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device, self.dtype)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(self.device, self.dtype)
        self.scheduler = LCMScheduler.from_pretrained(model_key, subfolder="scheduler")

        # Freeze models
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval()
        print(f"Loaded LCM!")

    def get_text_embeds(self, prompt):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_embeddings

    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def get_w_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        see https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298
        Args:
        timesteps: torch.Tensor: generate embedding vectors at these timesteps
        embedding_dim: int: dimension of the embeddings to generate
        dtype: data type of the generated embeddings
        Returns:
        embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    def set_attn_processor_mad(self, processor, block_name='all'):

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.processor.apply_mad = True
                else:
                    raise NotImplementedError

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        if 'down_blocks' in block_name:
            for name, module in self.unet.down_blocks.named_children():
                fn_recursive_attn_processor(name, module, processor)
        elif 'mid_block' in block_name:
            for name, module in self.unet.mid_block.named_children():
                fn_recursive_attn_processor(name, module, processor)
        elif 'up_blocks' in block_name:
            for name, module in self.unet.up_blocks.named_children():
                fn_recursive_attn_processor(name, module, processor)
        else:
            for name, module in self.unet.named_children():
                fn_recursive_attn_processor(name, module, processor)

    @torch.no_grad()
    def sample(
            self,
            prompts,
            height=512,
            width=3072,
            latent_size=64,  # fix latent size to 64 for Stable Diffusion
            num_inference_steps=4,
            guidance_scale=7.5,
            stride=16,  # stride for latents, set as 16 in the paper
            mad_blocks='all',
            mad_threshold=50
    ):

        if isinstance(prompts, str):
            prompts = [prompts]

        # obtain text embeddings
        text_embeds = self.get_text_embeds(prompts).to(self.dtype)  # [2, 77, 768]

        # define a list of windows to process in parallel
        views = get_views(height, width, window_size=latent_size, stride=stride)

        # Initialize latent
        latent = torch.randn((1, self.unet.config.in_channels, height // 8,  width // 8), dtype=self.dtype)

        processor = CrossViewsAttnProcessor2_0(
            batch_size=1,
            latent_h=height // 8,
            views=views,
            latent_w=width // 8,
            stride=stride,
            is_cons=True)


        self.unet.set_attn_processor(processor)
        self.set_attn_processor_mad(processor, mad_blocks)

        count = torch.zeros_like(latent, requires_grad=False, device=self.device, dtype=self.dtype)
        value_latent = torch.zeros_like(latent, requires_grad=False, device=self.device, dtype=self.dtype)
        latent = latent.to(self.device)

        lcm_origin_steps = 50
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps, original_inference_steps=lcm_origin_steps)

        b, l, d = text_embeds.size()
        text_embeds = text_embeds[:, None].repeat(1, len(views), 1, 1).reshape(b * len(views), l, d)
        w = torch.tensor(guidance_scale).repeat(len(views))
        w_embedding = self.get_w_embedding(w, embedding_dim=256).to(self.device)

        with torch.autocast('cuda'):
            for i, t in enumerate(tqdm(self.scheduler.timesteps)):
                if i >= mad_threshold:
                    processor.apply_mad = False

                count.zero_()
                value_latent.zero_()

                batched_latent_views = []
                for view_idx, (h_start, h_end, w_start, w_end) in enumerate(views):
                    latent_view = latent[:, :, h_start:h_end, w_start:w_end].detach()
                    batched_latent_views.append(latent_view)

                latent_model_input = torch.cat(batched_latent_views)
                noise_pred = self.unet(latent_model_input, t, timestep_cond=w_embedding, encoder_hidden_states=text_embeds)['sample']

                for view_idx, (h_start, h_end, w_start, w_end) in enumerate(views):
                    value_latent[:, :, h_start:h_end, w_start:w_end] += noise_pred[view_idx]
                    count[:, :, h_start:h_end, w_start:w_end] += 1

                noise_pred = torch.where(count > 0, value_latent / count, value_latent)
                latent, denoised = self.scheduler.step(noise_pred, t, latent, return_dict=False)
                latent = latent.to(self.dtype)

        # decode latents to panorama image
        with torch.no_grad():
            imgs = self.decode_latents(denoised)
            img = T.ToPILImage()(imgs[0].cpu())

        return img
