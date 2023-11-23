import torch
import torch.nn as nn
import torchvision.transforms as T
from tqdm import tqdm

from sampling_utils import get_views
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from nn_utils import CrossViewsAttnProcessor2_0


class MergeAttendStableDiffusion(nn.Module):
    def __init__(self, model_key, device='cuda', dtype='fp32'):
        super().__init__()

        self.device = device
        self.dtype = torch.float16 if dtype == 'fp16' else torch.float32

        print('Loading Stable Diffusion...')

        # Load pretrained models from HuggingFace
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.device, self.dtype)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_enc = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device, self.dtype)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(self.device, self.dtype)

        # Freeze models
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.text_enc.parameters():
            p.requires_grad_(False)

        self.unet.eval()
        self.vae.eval()
        self.text_enc.eval()

        # Set scheduler
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        print(f"Loaded stable diffusion!")

    def get_text_embeds(self, prompt, negative_prompt):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_enc(text_input.input_ids.to(self.device))[0]

        # Repeat for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')
        uncond_embeddings = self.text_enc(uncond_input.input_ids.to(self.device))[0]

        # Concatenate for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

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
            negative_prompts="",
            height=512,
            width=3072,
            latent_size=64,  # fix latent size to 64 for Stable Diffusion
            num_inference_steps=50,
            guidance_scale=7.5,
            stride=16,  # stride for latents, set as 16 in the paper
            mad_blocks='all',
            mad_threshold=50
    ):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # obtain text embeddings
        text_embeds = self.get_text_embeds(prompts, negative_prompts).to(self.dtype)  # [2, 77, 768]

        # define a list of windows to process in parallel
        views = get_views(height, width, window_size=latent_size, stride=stride)

        # Initialize latent
        latent = torch.randn((1, self.unet.config.in_channels, height // 8, width // 8), dtype=self.dtype)

        processor = CrossViewsAttnProcessor2_0(
            batch_size=2,
            latent_h=height // 8,
            views=views,
            latent_w=width // 8,
            stride=stride,
            is_cons=False)

        self.unet.set_attn_processor(processor)
        self.set_attn_processor_mad(processor, mad_blocks)

        count = torch.zeros_like(latent, requires_grad=False, device=self.device, dtype=self.dtype)
        value = torch.zeros_like(latent, requires_grad=False, device=self.device, dtype=self.dtype)
        latent = latent.to(self.device)

        # set scheduler
        self.scheduler.set_timesteps(num_inference_steps)

        b, l, d = text_embeds.size()
        text_embeds = text_embeds[:, None].repeat(1, len(views), 1, 1).reshape(b * len(views), l, d)

        with torch.autocast('cuda'):
            for i, t in enumerate(tqdm(self.scheduler.timesteps)):
                if i >= mad_threshold:
                    processor.apply_mad = False

                count.zero_()
                value.zero_()

                batched_latent_views = []
                for view_idx, (h_start, h_end, w_start, w_end) in enumerate(views):
                    latent_view = latent[:, :, h_start:h_end, w_start:w_end].detach()
                    batched_latent_views.append(latent_view)

                # expand the latents for classifier-free guidance
                latent_model_input = torch.cat(batched_latent_views * 2)
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds)['sample']
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred_new = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latent_view_dn = self.scheduler.step(noise_pred_new, t, torch.cat(batched_latent_views))['prev_sample']

                for view_idx, (h_start, h_end, w_start, w_end) in enumerate(views):
                    value[:, :, h_start:h_end, w_start:w_end] += latent_view_dn[view_idx]
                    count[:, :, h_start:h_end, w_start:w_end] += 1

                # take the MultiDiffusion step (average the latents)
                latent = torch.where(count > 0, value / count, value)

        # decode latents to panorama image
        with torch.no_grad():
            imgs = self.decode_latents(latent)
            img = T.ToPILImage()(imgs[0].cpu())

        return img
