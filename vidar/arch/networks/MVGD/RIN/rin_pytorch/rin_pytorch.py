import math
from functools import partial
import clip

import torch
from torch import nn
import torch.nn.functional as F

from timm.models.layers import drop_path

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from vidar.utils.types import is_dict, is_list
from vidar.utils.data import make_list, first_value

import xformers.ops as xops


# helpers functions
def exists(x):
    return x is not None

def identity(x):
    return x

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0

def safe_div(numer, denom, eps = 1e-10):
    return numer / denom.clamp(min = eps)

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def Sequential(*mods):
    return nn.Sequential(*filter(exists, mods))

# use layernorm without bias, more stable

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# positional embeds

class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        norm = False,
        time_cond_dim = None
    ):
        super().__init__()
        hidden_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.time_cond = None

        if exists(time_cond_dim):
            self.time_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim * 2),
                Rearrange('b d -> b 1 d')
            )

            nn.init.zeros_(self.time_cond[-2].weight)
            nn.init.zeros_(self.time_cond[-2].bias)

        self.norm = LayerNorm(dim) if norm else nn.Identity()

        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim, bias = False),
            LayerNorm(dim)
        )

    def forward(
        self,
        x,
        time = None
    ):
        h = self.heads
        x = self.norm(x)

        if exists(self.time_cond):
            assert exists(time)
            scale, shift = self.time_cond(time).chunk(2, dim = -1)
            x = (x * (scale + 1)) + shift

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        context = torch.einsum('b h n d, b h n e -> b h d e', k, v)

        out = torch.einsum('b h d e, b h n d -> b h n e', context, q)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_context = None,
        heads = 4,
        dim_head = 32,
        norm = False,
        norm_context = False,
        drop_path=0.0
    ):
        super().__init__()
        hidden_dim = dim_head * heads
        dim_context = default(dim_context, dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.norm = LayerNorm(dim) if norm else nn.Identity()
        self.norm_context = LayerNorm(dim_context) if norm_context else nn.Identity()
        self.dropp = DropPath(drop_path)

        self.to_q = nn.Linear(dim, hidden_dim, bias = False)
        self.to_kv = nn.Linear(dim_context, hidden_dim * 2, bias = False)
        self.to_out = nn.Linear(hidden_dim, dim, bias = False)

    def forward(self, x, context=None):

        h = self.heads
        if exists(context):
            context = self.norm_context(context)
        x_res = self.norm(x)

        context = default(context, x_res)
        qkv = (self.to_q(x_res), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h = h), qkv)

        out = xops.memory_efficient_attention(q, k, v, p=0.0, scale=self.scale).flatten(-2)
        # out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        # out = rearrange(out, 'b h n d -> b n h d').flatten(-2)

        out = self.to_out(out)

        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, drop_units=0.1, drop_path=0.0, pre_ln=True):
        super().__init__()
        self.norm = None
        if pre_ln:
            self.norm = LayerNorm(dim)

        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(drop_units),
            nn.Linear(inner_dim, dim)
        )
        
        self.dropp = DropPath(drop_path)

    def forward(self, x):
        if self.norm is not None:
            x_residual = self.norm(x)
        else:
            x_residual = x

        x_residual = self.net(x_residual)
        return x_residual

# model
class RINBlock(nn.Module):
    def __init__(
        self,
        dim,
        latent_self_attn_depth,
        dim_latent = None,
        final_norm = True,
        rw_heads = 8,
        latent_heads = 16,
        drop_path=0.0,
        **attn_kwargs
    ):
        super().__init__()
        dim_latent = default(dim_latent, dim)

        self.latents_attend_to_patches = Attention(
            dim_latent, dim_context=dim, norm=True, norm_context=False, 
            heads=rw_heads, drop_path=0, **attn_kwargs)
        self.latents_cross_attn_ff = FeedForward(dim_latent, drop_path=0)

        self.latent_self_attns = nn.ModuleList([])
        for _ in range(latent_self_attn_depth):
            self.latent_self_attns.append(nn.ModuleList([
                Attention(dim_latent, norm=True, heads=latent_heads, drop_path=drop_path, **attn_kwargs),
                FeedForward(dim_latent, drop_path=drop_path)
            ]))

        self.patches_attend_to_latents = Attention(
            dim, dim_context=dim_latent, norm=True, norm_context=False, 
            heads=rw_heads, drop_path=0, **attn_kwargs)
        self.patches_cross_attn_ff = FeedForward(dim, drop_path=0)

    def forward(self, patches, latents):

        latents = self.latents_attend_to_patches(latents, patches) + latents
        latents = self.latents_cross_attn_ff(latents) + latents

        for attn, ff in self.latent_self_attns:
            latents = attn(latents) + latents
            latents = ff(latents) + latents

        patches = self.patches_attend_to_latents(patches, latents) + patches
        patches = self.patches_cross_attn_ff(patches) + patches

        return patches, latents

class RIN(nn.Module):
    def __init__(self, cfg, lengths):
        super().__init__()

        self.lengths = lengths
        self.channels = cfg.has('channels', None)
        self.num_latents = cfg.num_latents
        self.layers2pixels = cfg.has('layers2pixels', None)
        self.layers2patches = cfg.has('layers2patches', None)
        self.with_language_conditioning = cfg.has('language_conditioning', False)
        self.preserve_encode = cfg.has('preserve_encode', False)

        fourier_dim = cfg.learned_sinusoidal_dim + 1
        dim_latent = cfg.has('dim_latent', cfg.dim)
        time_dim = cfg.dim * 4

        self.tasks = cfg.has('tasks', None)
        self.tasks_encode_idx =  cfg.has('tasks_encode_idx', None)
        self.tasks_decode_idx =  cfg.has('tasks_decode_idx', None)

        if cfg.has('tasks_dim'):
            if is_list(cfg.decode_dim):
                for i in range(len(cfg.decode_dim)):
                    cfg.decode_dim[i] += cfg.tasks_dim
            else:
                cfg.decode_dim += cfg.tasks_dim

        self.time_mlp = nn.Sequential(
            LearnedSinusoidalPosEmb(cfg.learned_sinusoidal_dim),
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, dim_latent)
        )

        if self.with_language_conditioning:
            self.language_mlp = nn.Sequential(
                nn.Linear(self.with_language_conditioning, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, dim_latent)
            )
        else:
            self.language_mlp = None

        if self.tasks is None:
            self.to_patches2 = self.make_to_patches(cfg.dim, cfg.decode_dim)
        else:
            if is_list(cfg.decode_dim):
                self.to_patches2 = nn.ModuleDict()
                for i, task in enumerate(self.tasks):
                    self.to_patches2[task] = self.make_to_patches(
                        cfg.dim, cfg.decode_dim[self.tasks_encode_idx[i]], task)
            else:
                self.to_patches2 = nn.ModuleDict()
                for task in self.tasks:
                    self.to_patches2[task] = self.make_to_patches(
                        cfg.dim, cfg.decode_dim, task)

        if is_list(cfg.encode_dim):
            encoder_emb_mlp = nn.ModuleList([Sequential(
                nn.Linear(encode_dim, cfg.dim),
            ) for encode_dim in cfg.encode_dim])
            self.encoder_emb_mlp = nn.ModuleDict({key: encoder_emb_mlp[val] 
                for key, val in zip(self.tasks, self.tasks_encode_idx)})
        else:
            self.encoder_emb_mlp = Sequential(
                nn.Linear(cfg.encode_dim, cfg.dim),
            )

        if self.tasks is None:
            self.to_pixels2 = self.make_to_pixels(cfg.dim)
        else:
            self.to_pixels2 = nn.ModuleDict()
            for task in self.tasks:
                self.to_pixels2[task] = self.make_to_pixels(cfg.dim, task)

        self.num_latents = cfg.num_latents
        self.latents = nn.Parameter(torch.randn(cfg.num_latents, dim_latent))
        nn.init.normal_(self.latents, std=0.02)

        self.init_self_cond_latents = nn.Sequential(
            FeedForward(dim_latent, pre_ln=False, drop_path=0.0),
            LayerNorm(dim_latent),
        )
        nn.init.zeros_(self.init_self_cond_latents[-1].gamma)

        self.blocks = nn.ModuleList([RINBlock(
            cfg.dim, 
            dim_latent=dim_latent, 
            latent_self_attn_depth=cfg.latent_self_attn_depth, 
            rw_heads=cfg.rw_heads, 
            latent_heads=cfg.latent_heads, 
            drop_path=0.0, 
        ) for _ in range(cfg.depth)])

        if cfg.has('tasks'):
            self.task_embeddings = nn.ParameterDict()
            for task in cfg.tasks:
                # self.task_embeddings[task] = nn.Parameter(torch.randn(1, 1, 512)) # 8)) # 32)) # 128))
                self.task_embeddings[task] = nn.Parameter(torch.randn(1, 1, cfg.tasks_dim))
                nn.init.normal_(self.task_embeddings[task], std=0.02)
        else:
            self.task_embeddings = None
                
    @property
    def device(self):
        return next(self.parameters()).device
    
    def make_to_pixels(self, dim, task=None):
        if self.layers2pixels is None:
            return Sequential(
                LayerNorm(dim),
                nn.Linear(dim, self.task_channels(task)),
            )
        else:
            return Sequential(
                *([LayerNorm(dim),
                   nn.Linear(dim, dim)] * self.layers2pixels),
                LayerNorm(dim),
                nn.Linear(dim, self.task_channels(task)),
            )

    def make_to_patches(self, dim, decode_dim, task=None):
        if self.layers2patches is None:
            return Sequential(
                nn.Linear(self.task_channels(task) + decode_dim, dim),
                nn.LayerNorm(dim),
            )
        else:
            return Sequential(
                nn.Linear(self.task_channels(task) + decode_dim, dim),
                nn.LayerNorm(dim),
                *([LayerNorm(dim),
                   nn.Linear(dim, dim)] * self.layers2pixels),
            )

    def task_channels(self, task):
        if task is None:
            return self.channels
        else:
            return self.lengths[task]
    
    def timestep_conditioning_fn(self, timestep, latents, latent_self_cond, 
                                 initialized):
        timestep = self.time_mlp(timestep)
        timestep = rearrange(timestep, 'b d -> b 1 d')

        if not initialized:
            latent_self_cond = torch.cat((latent_self_cond, timestep), dim=-2)
        latents = torch.cat((latents, timestep), dim=-2)
        return latents, latent_self_cond

    def language_conditioning_fn(self, language, latents, latent_self_cond,
                                 initialized):
        if language is None or not self.with_language_conditioning:
            return latents, latent_self_cond

        embeddings = first_value(language)['embeddings']
        language = self.language_mlp(embeddings)

        if not initialized:
            latent_self_cond = torch.cat((latent_self_cond, language), dim=-2)
        latents = torch.cat((latents, language), dim=-2)

        return latents, latent_self_cond

    def encoder_conditioning_fn(self, embeddings, latents, latent_self_cond, 
                                initialized):
        embeddings = self.encoder_emb_mlp(embeddings)

        if not initialized:
            latent_self_cond = torch.cat((latent_self_cond, embeddings), dim=-2)
        latents = torch.cat((latents, embeddings), dim=-2)
        return latents, latent_self_cond

    def forward(self, x, timestep, embeddings,
                latent_self_cond=None, return_latents=False,
                cond_drop_prob=None, task=None, idx=None, language=None):

        # Check if input is an image
        is_image = x.dim() == 4

        # Flatten image if required
        if is_image:
            b, c, h, w = x.shape
            x = x.view(*x.shape[:2], -1).permute(0, 2, 1)
        else:
            b = x.shape[0]

        # Repeat latents batch-wise
        latents = repeat(self.latents, 'n d -> b n d', b=b)

        # Initialize parameters
        initialized = latent_self_cond is not None
        cond_drop_prob = default(cond_drop_prob, 0.0)
        latent_self_cond = default(latent_self_cond, torch.zeros_like(latents))

        # Get embeddings
        encode_embeddings, decode_embeddings = embeddings
        if is_dict(encode_embeddings):
            if len(encode_embeddings.values()) > 0:
                encode_embeddings = torch.cat([val for val in encode_embeddings.values()], 1)
            else:
                encode_embeddings = None
        if is_dict(decode_embeddings):
            decode_embeddings = torch.cat([val for val in decode_embeddings.values()], 1)

        if task is not None:
            # if task not in ['action']:
            decode_embeddings = torch.cat([
                decode_embeddings, self.task_embeddings[task].repeat(
                    *decode_embeddings.shape[:2], 1)], -1)
            # else:
            #     b, n = x.shape[:2]
            #     decode_embeddings = self.task_embeddings[task].repeat(*x.shape[:2], 1)

        # Timestep conditioning
        latents, latent_self_cond = self.timestep_conditioning_fn(
            timestep, latents, latent_self_cond, initialized,
        )

        # Language conditioning
        latents, latent_self_cond = self.language_conditioning_fn(
            language, latents, latent_self_cond, initialized,
        )

        # Self-conditioning
        latents = latents + self.init_self_cond_latents(latent_self_cond)

        # Local conditioning
        patches = x.view(*x.shape[:2], -1)
        patches = torch.cat([patches, decode_embeddings], 2)

        if self.packer is None:
            to_patches2 = self.to_patches2 if not is_dict(self.to_patches2) else self.to_patches2[task]
            to_patches2 = to_patches2 if not is_list(to_patches2) else to_patches2[0]
            patches = to_patches2(patches)
        else:
            to_patches2 = self.to_patches2 if not is_dict(self.to_patches2) else self.to_patches2[task]
            to_patches2 = to_patches2 if not is_list(to_patches2) else to_patches2[0]
            patches = to_patches2(patches)
            patches = self.packer.pack(patches)

        # Number of local tokens
        n = patches.shape[1]

        # Global conditioning
        encoder_emb_mlp = self.encoder_emb_mlp if not is_dict(self.encoder_emb_mlp) else self.encoder_emb_mlp[task]
        encode_embeddings = encoder_emb_mlp(encode_embeddings.float()) if encode_embeddings is not None else None

        if not self.preserve_encode and encode_embeddings is not None:
            patches = torch.cat([patches, encode_embeddings], 1)

        # RIN processing
        for _, block in enumerate(self.blocks):
            if self.preserve_encode and encode_embeddings is not None:
                patches = torch.cat([patches, encode_embeddings], 1)
            patches, latents = block(patches, latents)
            if self.preserve_encode and encode_embeddings is not None:
                patches = patches[:, :n]
            
            # U, S, V = torch.pca_lowrank(patches.float())
            # aaa = torch.matmul(V.permute(0, 2, 1)[:, :3], patches.permute(0, 2, 1))
            # print('qwerqwerq', aaa.shape)
            # aaa = aaa[:, :, :n]
            # print('qwerqwerqwr', aaa.shape)
            # aaa = aaa.view(1, 3, 120, 160)
            # aaa = (aaa - (aaa.min())) / (aaa.max() - aaa.min())
            # from vidar.utils.write import write_image
            # write_image('pca__%03d_%d.png' % (i, timestep[0].cpu().numpy()), aaa)
            # print(aaa.shape)

        # Keep only local vectors
        if not self.preserve_encode and encode_embeddings is not None:
            patches = patches[:, :n]

        # Decode from tokens
        if self.packer is None:
            pixels = self.to_pixels2(patches) if not is_dict(self.to_pixels2) else self.to_pixels2[task](patches)
            invalid = torch.zeros_like(pixels).bool()
        else:
            patches, invalid = self.packer.unpack(patches)
            pixels = self.to_pixels2(patches) if not is_dict(self.to_pixels2) else self.to_pixels2[task](patches)

        # Back to image shape if required
        if is_image:
            pixels = pixels.permute(0, 2, 1).view(b, c, h, w)

        # Return values
        if return_latents:
            return pixels, latents
        else:
            return pixels, invalid

