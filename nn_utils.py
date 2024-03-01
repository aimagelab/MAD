import torch
import torch.nn.functional as F
import math
from einops import rearrange


class CrossViewsAttnProcessor2_0:
    """
    Cross frame attention processor with scaled_dot_product attention of Pytorch 2.0.
    """

    def __init__(self, latent_h, latent_w, views, batch_size=1, stride=16, latent_size=64, mad=False, is_cons=False):

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.latent_h = latent_h
        self.latent_w = latent_w
        self.views = views
        self.bs = batch_size
        self.stride = stride
        self.mad = mad
        self.is_cons = is_cons
        self.latent_size = latent_size

    def compute_current_sizes(self, batch):
        bs, sequence_length, inner_dim = batch.shape
        views_len = bs // self.bs
        spatial_size = int(math.sqrt(sequence_length))
        down_factor = self.latent_size // spatial_size
        latent_h = self.latent_h // down_factor
        latent_w = self.latent_w // down_factor
        return views_len, spatial_size, down_factor, latent_h, latent_w, inner_dim

    def merge_all_batched_qkv_views_into_canvas(self, batch_q, batch_k, batch_v):
        views_len, spatial_size, down_factor, latent_h, latent_w, inner_dim = self.compute_current_sizes(batch_q)
        batch_q_views = rearrange(batch_q, '(b v) (h w) d -> b v d h w', v=views_len, h=spatial_size)
        batch_k_views = rearrange(batch_k, '(b v) (h w) d -> b v d h w', v=views_len, h=spatial_size)
        batch_v_views = rearrange(batch_v, '(b v) (h w) d -> b v d h w', v=views_len, h=spatial_size)
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
        batch_views = rearrange(batch, '(b v) (h w) d -> b v d h w', v=views_len, h=spatial_size)
        canvas = torch.zeros((self.bs, inner_dim, latent_h, latent_w), device=self.device, dtype=self.dtype)
        count = torch.zeros((self.bs, inner_dim, latent_h, latent_w), device=self.device, dtype=self.dtype)
        for view_idx, (h_start, h_end, w_start, w_end) in enumerate(self.views):
            h_start, h_end = h_start // down_factor, h_end // down_factor
            w_start, w_end = w_start // down_factor, w_end // down_factor
            canvas[:, :, h_start:h_end, w_start:w_end] += batch_views[:, view_idx]
            count[:, :, h_start:h_end, w_start:w_end] += 1
        batch = torch.where(count > 0, canvas / count, canvas)
        return batch, down_factor

    def split_canvas_into_views(self, canvas, down_factor):
        canvas_views = []
        for view_idx, (h_start, h_end, w_start, w_end) in enumerate(self.views):
            h_start, h_end = h_start // down_factor, h_end // down_factor
            w_start, w_end = w_start // down_factor, w_end // down_factor
            canvas_views.append(canvas[:, :, h_start:h_end, w_start:w_end, :])
        canvas = torch.cat(canvas_views, dim=1)
        canvas = rearrange(canvas, 'b v h w d -> (b v) (h w) d')
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

        if not is_cross_attention:
            if self.apply_mad:
                query, key, value, down_factor = self.merge_all_batched_qkv_views_into_canvas(query, key, value)
                query = rearrange(query, 'b (nh hd) h w -> b nh (h w) hd', nh=attn.heads, hd=head_dim).contiguous()
                key = rearrange(key, 'b (nh hd) h w -> b nh (h w) hd', nh=attn.heads, hd=head_dim).contiguous()
                value = rearrange(value, 'b (nh hd) h w -> b nh (h w) hd', nh=attn.heads, hd=head_dim).contiguous()
            else:
                query = rearrange(query, 'b hw (nh nd) -> b nh hw nd', nh=attn.heads, nd=head_dim)
                key = rearrange(key, 'b hw (nh nd) -> b nh hw nd', nh=attn.heads, nd=head_dim)
                value = rearrange(value, 'b hw (nh nd) -> b nh hw nd', nh=attn.heads, nd=head_dim)
        else:
            query, down_factor = self.merge_batched_q_views_into_canvas(query)
            query = rearrange(query, 'b (nh hd) h w -> b nh (h w) hd', nh=attn.heads, hd=head_dim)
            # LCMs do not allow negative prompts
            key = torch.cat([key[None, 0], key[None, -1]], dim=0) if not self.is_cons else key[None, 0]
            key = rearrange(key, 'b p (nh nd) -> b nh p nd', nh=attn.heads, nd=head_dim)
            value = torch.cat([value[None, 0], value[None, -1]], dim=0) if not self.is_cons else value[None, 0]
            value = rearrange(value, 'b p (nh nd) -> b nh p nd', nh=attn.heads, nd=head_dim)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False, scale=attn.scale
        )

        if not is_cross_attention and not self.apply_mad:
            hidden_states = rearrange(hidden_states, 'b nh hw nd -> b hw (nh nd)')
        else:
            hidden_states = hidden_states.transpose(1, 2)
            latent_h = self.latent_h // down_factor
            hidden_states = rearrange(hidden_states, 'b (h w) nh hd -> b 1 h w (nh hd)', h=latent_h)
            hidden_states = self.split_canvas_into_views(hidden_states, down_factor)

        hidden_states = hidden_states.to(query.dtype)
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
