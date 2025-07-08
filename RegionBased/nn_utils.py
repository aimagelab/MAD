import torch
import torch.nn.functional as F
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path


class CrossFrameAttnProcessor2_0:
    """
    Cross frame attention processor with scaled_dot_product attention of Pytorch 2.0.

    Args:
        batch_size: The number that represents actual batch size, other than the frames.
            For example, calling unet with a single prompt and num_images_per_prompt=1, batch_size should be equal to
            2, due to classifier-free guidance.
    """

    def __init__(self, latent_h, latent_w, views, batch_size=1, stride=16, ca_type="merged", sa_type="batch",
                 is_cons=False, activate_region_based_guidance=False):

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        if ca_type not in ["batch", "merged", "split"]:
            raise ValueError(f"ca_type must be one of ['batch', 'merged', 'split'], got {ca_type}")
        if sa_type not in ["batch", "merged", "split"]:
            raise ValueError(f"sa_type must be one of ['batch', 'merged', 'split'], got {sa_type}")

        self.latent_h = latent_h
        self.latent_w = latent_w
        self.views = views
        self.bs = batch_size
        self.stride = stride
        self.cross_attention_type = ca_type
        self.self_attention_type = sa_type
        self.is_cons = is_cons

        self.current_region_mask = None
        self.activate_region_based_guidance = activate_region_based_guidance


    def compute_current_sizes(self, batch):
        bs, sequence_length, inner_dim = batch.shape
        views_len = bs // self.bs
        spatial_size = int(math.sqrt(sequence_length))
        down_factor = 64 // spatial_size
        latent_h = self.latent_h // down_factor
        latent_w = self.latent_w // down_factor
        return views_len, spatial_size, down_factor, latent_h, latent_w, inner_dim


    def merge_all_batched_qkv_views_into_canvas(self, batch_q, batch_k, batch_v):
        views_len, spatial_size, down_factor, latent_h, latent_w, inner_dim = self.compute_current_sizes(batch_q)
        batch_q_views = batch_q.reshape(self.bs, views_len, spatial_size, spatial_size, -1).permute(0, 1, 4, 2, 3)
        batch_k_views = batch_k.reshape(self.bs, views_len, spatial_size, spatial_size, -1).permute(0, 1, 4, 2, 3)
        batch_v_views = batch_v.reshape(self.bs, views_len, spatial_size, spatial_size, -1).permute(0, 1, 4, 2, 3)
        canvas_q = torch.zeros((self.bs, inner_dim, latent_h, latent_w), device=self.device, dtype=self.dtype)
        canvas_k = torch.zeros((self.bs, inner_dim, latent_h, latent_w), device=self.device, dtype=self.dtype)
        canvas_v = torch.zeros((self.bs, inner_dim, latent_h, latent_w), device=self.device, dtype=self.dtype)
        count = torch.zeros((self.bs, inner_dim, latent_h, latent_w), device=self.device, dtype=self.dtype)
        for view_idx, (h_start, h_end, w_start, w_end) in enumerate(self.views):
            h_start, h_end = h_start // down_factor, h_end // down_factor
            w_start, w_end = w_start // down_factor, w_end // down_factor
            canvas_q[:, :, h_start:h_end, w_start:w_end] += batch_q_views[:, view_idx]
            canvas_k[:, :, h_start:h_end, w_start:w_end] += batch_k_views[:, view_idx]
            canvas_v[:, :, h_start:h_end, w_start:w_end] += batch_v_views[:, view_idx]
            count[:, :, h_start:h_end, w_start:w_end] += 1
        batch_q = torch.where(count > 0, canvas_q / count, canvas_q)
        batch_k = torch.where(count > 0, canvas_k / count, canvas_k)
        batch_v = torch.where(count > 0, canvas_v / count, canvas_v)
        return batch_q, batch_k, batch_v, down_factor


    def merge_batched_q_views_into_canvas(self, batch):
        views_len, spatial_size, down_factor, latent_h, latent_w, inner_dim = self.compute_current_sizes(batch)
        batch_views = batch.reshape(self.bs, views_len, spatial_size, spatial_size, -1).permute(0, 1, 4, 2, 3)
        canvas = torch.zeros((self.bs, inner_dim, latent_h, latent_w), device=self.device, dtype=self.dtype)
        count = torch.zeros((self.bs, inner_dim, latent_h, latent_w), device=self.device, dtype=self.dtype)
        for view_idx, (h_start, h_end, w_start, w_end) in enumerate(self.views):
            h_start, h_end = h_start // down_factor, h_end // down_factor
            w_start, w_end = w_start // down_factor, w_end // down_factor
            canvas[:, :, h_start:h_end, w_start:w_end] += batch_views[:, view_idx]
            count[:, :, h_start:h_end, w_start:w_end] += 1
        batch = torch.where(count > 0, canvas / count, canvas)
        return batch, down_factor


    def split_qkv_canvas_into_views(self, canvas_q, canvas_k, canvas_v, down_factor):
        canvas_q_views, canvas_k_views, canvas_v_views  = [], [], []
        for view_idx, (h_start, h_end, w_start, w_end) in enumerate(self.views):
            h_start, h_end = h_start // down_factor, h_end // down_factor
            w_start, w_end = w_start // down_factor, w_end // down_factor
            canvas_q_views.append(canvas_q[:, :, h_start:h_end, w_start:w_end, :])
            canvas_k_views.append(canvas_k[:, :, h_start:h_end, w_start:w_end, :])
            canvas_v_views.append(canvas_v[:, :, h_start:h_end, w_start:w_end, :])
        canvas_q = torch.cat(canvas_q_views, dim=1)
        canvas_k = torch.cat(canvas_k_views, dim=1)
        canvas_v = torch.cat(canvas_v_views, dim=1)
        batch_size, views_len, height, width, inner_dim = canvas_q.size()
        canvas_q = canvas_q.reshape(batch_size * views_len, height * width, inner_dim)
        canvas_k = canvas_k.reshape(batch_size * views_len, height * width, inner_dim)
        canvas_v = canvas_v.reshape(batch_size * views_len, height * width, inner_dim)
        return canvas_q, canvas_k, canvas_v


    def split_canvas_into_views(self, canvas, down_factor):
        canvas_views = []
        for view_idx, (h_start, h_end, w_start, w_end) in enumerate(self.views):
            h_start, h_end = h_start // down_factor, h_end // down_factor
            w_start, w_end = w_start // down_factor, w_end // down_factor
            canvas_views.append(canvas[:, :, h_start:h_end, w_start:w_end, :])
        canvas = torch.cat(canvas_views, dim=1)
        batch_size, views_len, height, width, inner_dim = canvas.size()
        canvas = canvas.reshape(batch_size * views_len, height * width, inner_dim)
        return canvas


    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        bs, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        inner_dim = hidden_states.shape[-1]
        self.device = hidden_states.device
        self.dtype = hidden_states.dtype

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, bs)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(bs, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        head_dim = inner_dim // attn.heads

        if is_cross_attention:
            if self.cross_attention_type == "merged":
                query, down_factor = self.merge_batched_q_views_into_canvas(query)
                canvas_height, canvas_width = query.shape[-2], query.shape[-1]
                query = query.reshape(self.bs, attn.heads, head_dim, -1).permute(0, 1, 3, 2).contiguous()
                key = torch.cat([key[None, 0], key[None, -1]], dim=0) if not self.is_cons else key[None, 0]
                key = key.view(self.bs, -1, attn.heads, head_dim).transpose(1, 2)
                value = torch.cat([value[None, 0], value[None, -1]], dim=0) if not self.is_cons else value[None, 0]
                value = value.view(self.bs, -1, attn.heads, head_dim).transpose(1, 2)
            else:
                if self.cross_attention_type == 'split':
                    query, down_factor = self.merge_batched_q_views_into_canvas(query)
                    canvas_height, canvas_width = query.shape[-2], query.shape[-1]
                    query = query.permute(0, 2, 3, 1).unsqueeze(1)
                    query = self.split_canvas_into_views(query, down_factor)
                query = query.view(bs, -1, attn.heads, head_dim).transpose(1, 2)
                key = key.view(bs, -1, attn.heads, head_dim).transpose(1, 2)
                value = value.view(bs, -1, attn.heads, head_dim).transpose(1, 2)
        else:
            if self.self_attention_type == 'merged':
                query, key, value, down_factor = self.merge_all_batched_qkv_views_into_canvas(query, key, value)
                query = query.reshape(self.bs, attn.heads, head_dim, -1).permute(0, 1, 3, 2).contiguous()
                key = key.reshape(self.bs, attn.heads, head_dim, -1).permute(0, 1, 3, 2).contiguous()
                value = value.reshape(self.bs, attn.heads, head_dim, -1).permute(0, 1, 3, 2).contiguous()
            else:
                if self.self_attention_type == "split":
                    query, key, value, down_factor = self.merge_all_batched_qkv_views_into_canvas(query, key, value)
                    query = query.permute(0, 2, 3, 1).unsqueeze(1)
                    key = key.permute(0, 2, 3, 1).unsqueeze(1)
                    value = value.permute(0, 2, 3, 1).unsqueeze(1)
                    query, key, value = self.split_qkv_canvas_into_views(query, key, value, down_factor)
                query = query.view(bs, -1, attn.heads, head_dim).transpose(1, 2)
                key = key.view(bs, -1, attn.heads, head_dim).transpose(1, 2)
                value = value.view(bs, -1, attn.heads, head_dim).transpose(1, 2)

        if is_cross_attention:
            if self.cross_attention_type == "merged":
                query = query.reshape(self.bs * attn.heads, -1, head_dim)  # todo redo using the functions avoiding multiple reshapes
                key = key.reshape(self.bs * attn.heads, -1, head_dim)
                value = value.reshape(self.bs * attn.heads, -1, head_dim)
            else:
                query = query.reshape(bs * attn.heads, -1, head_dim)
                key = key.reshape(bs * attn.heads, -1, head_dim)
                value = value.reshape(bs * attn.heads, -1, head_dim)

            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            if self.activate_region_based_guidance:
                if self.cross_attention_type == 'batch':
                    _, _, down_factor, canvas_height, canvas_width, _ = self.compute_current_sizes(attention_probs)
                current_region_mask = F.interpolate(self.current_region_mask, size=(canvas_height, canvas_width), mode='bilinear', align_corners=False)
                if self.cross_attention_type == 'merged':
                    attention_probs = attention_probs.reshape(self.bs * attn.heads, canvas_height, canvas_width, sequence_length)
                    attention_probs = attention_probs.permute(0, 3, 1, 2).contiguous()
                    attention_probs = attention_probs * current_region_mask
                    attention_probs = attention_probs.permute(0, 2, 3, 1).contiguous()
                    attention_probs = attention_probs.reshape(self.bs * attn.heads, -1, sequence_length)
                else:
                    attention_probs = attention_probs.reshape(bs, attn.heads, canvas_height, canvas_height, sequence_length)
                    attention_probs = attention_probs.permute(0, 1, 4, 2, 3).contiguous()
                    for view_idx, (h_start, h_end, w_start, w_end) in enumerate(self.views):
                        h_start, h_end = h_start // down_factor, h_end // down_factor
                        w_start, w_end = w_start // down_factor, w_end // down_factor
                        attention_probs[view_idx] *= current_region_mask[:, :, h_start:h_end, w_start:w_end]
                        attention_probs[view_idx + len(self.views)] *= current_region_mask[:, :, h_start:h_end, w_start:w_end]
                    attention_probs = attention_probs.permute(0, 1, 3, 4, 2).contiguous()
                    attention_probs = attention_probs.reshape(bs * attn.heads, canvas_height * canvas_height, sequence_length)

            hidden_states = torch.bmm(attention_probs, value)
            if self.cross_attention_type == 'merged':
                hidden_states = hidden_states.reshape(self.bs, attn.heads, -1, head_dim)
            else:
                hidden_states = hidden_states.reshape(bs, attn.heads, -1, head_dim)
        else:
            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        attention_type = self.cross_attention_type if is_cross_attention else self.self_attention_type
        if attention_type in ["batch", "split"]:
            hidden_states = hidden_states.transpose(1, 2).reshape(bs, -1, attn.heads * head_dim)
        elif attention_type == "merged":
            hidden_states = hidden_states.transpose(1, 2)
            latent_h = self.latent_h // down_factor
            latent_w = self.latent_w // down_factor
            hidden_states = hidden_states.reshape(self.bs, 1, latent_h, latent_w, attn.heads * head_dim)
            hidden_states = self.split_canvas_into_views(hidden_states, down_factor)

        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states



class CrossAttnStoreProcessor:
    def __init__(self):
        self.cross_attention_maps = []
        self.inner_prompt_values = []

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, inner_dim = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        head_dim = inner_dim // attn.heads

        if is_cross_attention:
            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            self.cross_attention_maps.append(attention_probs.cpu())
            self.inner_prompt_values.append(value.cpu())
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)
        else:
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states