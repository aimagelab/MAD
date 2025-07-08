from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

import logging

import torch
import torch.nn as nn
import torchvision.transforms as T
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

from sampling_utils import seed_everything, get_views, preprocess_mask
from nn_utils import CrossFrameAttnProcessor2_0, CrossAttnStoreProcessor

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)



class GlobalAttentionRegion(nn.Module):
    def __init__(self, device='cuda', dtype='fp32'):
        super().__init__()

        self.device = device
        self.dtype = torch.float16 if dtype == 'fp16' else torch.float32

        logger.info('Loading Stable Diffusion...')
        model_key = "stabilityai/stable-diffusion-2-base"

        # Load pretrained models from HuggingFace
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.device).to(self.dtype)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device).to(self.dtype)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(self.device).to(self.dtype)

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")

        logger.info(f"Loaded stable diffusion!")

    @torch.no_grad()
    def get_random_background(self, n_samples, height, width):
        # sample random background with a constant rgb value
        backgrounds = torch.rand(n_samples, 3, device=self.device, dtype=self.dtype)[:, :, None, None].repeat(1, 1, height, width)
        return torch.cat([self.encode_imgs(bg.unsqueeze(0)) for bg in backgrounds])

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    @torch.no_grad()
    def encode_imgs(self, imgs):
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215
        return latents

    @torch.no_grad()
    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    @torch.no_grad()
    def set_attn_processor_sa_type(self, processor, block_name=['all'], sa_type='merged'):

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.processor.self_attention_type = sa_type
                else:
                    raise NotImplementedError

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        if 'all' in block_name:
            for name, module in self.unet.named_children():
                fn_recursive_attn_processor(name, module, processor)
        if 'down_blocks_all' in block_name:
            for name, module in self.unet.down_blocks.named_children():
                fn_recursive_attn_processor(name, module, processor)
        if 'down_blocks_0' in block_name:
            for name, module in self.unet.down_blocks[0].named_children():
                fn_recursive_attn_processor(name, module, processor)
        if 'down_blocks_1' in block_name:
            for name, module in self.unet.down_blocks[1].named_children():
                fn_recursive_attn_processor(name, module, processor)
        if 'down_blocks_2' in block_name:
            for name, module in self.unet.down_blocks[2].named_children():
                fn_recursive_attn_processor(name, module, processor)
        if 'down_blocks_3' in block_name:
            for name, module in self.unet.down_blocks[3].named_children():
                fn_recursive_attn_processor(name, module, processor)
        if 'mid_block' in block_name:
            for name, module in self.unet.mid_block.named_children():
                fn_recursive_attn_processor(name, module, processor)
        if 'up_blocks_all' in block_name:
            for name, module in self.unet.up_blocks.named_children():
                fn_recursive_attn_processor(name, module, processor)
        if 'up_blocks_0' in block_name:
            for name, module in self.unet.up_blocks[0].named_children():
                fn_recursive_attn_processor(name, module, processor)
        if 'up_blocks_1' in block_name:
            for name, module in self.unet.up_blocks[1].named_children():
                fn_recursive_attn_processor(name, module, processor)
        if 'up_blocks_2' in block_name:
            for name, module in self.unet.up_blocks[2].named_children():
                fn_recursive_attn_processor(name, module, processor)
        if 'up_blocks_3' in block_name:
            for name, module in self.unet.up_blocks[3].named_children():
                fn_recursive_attn_processor(name, module, processor)


    @torch.no_grad()
    def generate(
            self,
            masks,
            prompts,
            negative_prompts='',
            height=512,
            width=2048,
            num_inference_steps=50,
            guidance_scale=7.5,
            bootstrapping=20,
            stride=16,
            latent_size=64,
            where_global_self_attention=['mid_block'],
            global_self_attention_thres=50,
            ca_type='merged',
            sa_base_type='batch',
            sa_type='merged',
            disable_tqdm=False,
            activate_region_based_guidance=False
    ):

        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # get bootstrapping backgrounds
        # can move this outside of the function to speed up generation. i.e., calculate in init
        bootstrapping_backgrounds = self.get_random_background(bootstrapping, height, width)

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts)  # [2 * len(prompts), 77, 768]

        # Define panorama grid and get views
        latent = torch.randn((1, self.unet.config.in_channels, height // 8, width // 8), device=self.device, dtype=self.dtype)
        noise = latent.clone().repeat(len(prompts) - 1, 1, 1, 1)
        views = get_views(height, width, window_size=latent_size, stride=stride)
        count = torch.zeros_like(latent)
        value = torch.zeros_like(latent)

        self.scheduler.set_timesteps(num_inference_steps)
        latent_h = height // 8
        latent_w = width // 8

        processor = CrossFrameAttnProcessor2_0(
            batch_size=2,
            latent_h=latent_h,
            views=views,
            latent_w=latent_w,
            stride=stride,
            ca_type=ca_type,
            sa_type=sa_base_type,
            is_cons=False,
            activate_region_based_guidance=activate_region_based_guidance)

        self.unet.set_attn_processor(processor)
        self.set_attn_processor_sa_type(processor, where_global_self_attention, sa_type)

        for i, t in enumerate(tqdm(self.scheduler.timesteps, disable=disable_tqdm)):
            if i >= global_self_attention_thres:
                processor.self_attention_type = sa_base_type

            count.zero_()
            value.zero_()
            for p, prompt in enumerate(prompts):
                current_latent = latent.clone()
                if i < bootstrapping and p != 0:
                    bg = bootstrapping_backgrounds[torch.randint(0, bootstrapping, (1,))]
                    bg = self.scheduler.add_noise(bg, noise[p-1, None], t)
                    current_latent = current_latent * masks[p] + bg * (1 - masks[p])
                batched_latent_views = []
                batched_masks_views = []
                for view_idx, (h_start, h_end, w_start, w_end) in enumerate(views):
                    batched_latent_views.append(current_latent[:, :, h_start:h_end, w_start:w_end].detach())
                    batched_masks_views.append(masks[None, p, :, h_start:h_end, w_start:w_end].detach())
                # batched_latent_views = torch.cat(batched_latent_views, dim=0)
                # batched_masks_views = torch.cat(batched_masks_views, dim=0)

                processor.current_region_mask = masks[p, None].to(self.dtype)

                latent_model_input = torch.cat(batched_latent_views * 2).to(self.dtype)
                text_embeds_p_unc = text_embeds[p].repeat(len(views), 1, 1)
                text_embeds_p_cond = text_embeds[p + len(prompts)].repeat(len(views), 1, 1)
                text_embeds_p = torch.cat([text_embeds_p_unc, text_embeds_p_cond], dim=0)
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds_p)['sample']
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred_new = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latent_view_denoised = self.scheduler.step(noise_pred_new, t, torch.cat(batched_latent_views))['prev_sample']

                for view_idx, (h_start, h_end, w_start, w_end) in enumerate(views):
                    value[:, :, h_start:h_end, w_start:w_end] += latent_view_denoised[view_idx] * batched_masks_views[view_idx]
                    count[:, :, h_start:h_end, w_start:w_end] += batched_masks_views[view_idx]

            # take the MultiDiffusion step
            latent = torch.where(count > 0, value / count, value).to(self.dtype)

        # Img latents -> imgs
        imgs = self.decode_latents(latent)  # [1, 3, 512, 512]
        img = T.ToPILImage()(imgs[0].cpu())
        return img