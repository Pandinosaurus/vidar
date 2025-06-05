# Copyright 2025 Toyota Research Institute.  All rights reserved.

import torch
import numpy as np
import os

from vidar.arch.networks.MVGD.DepthAutoEncoder import LPIPSWithDiscriminator
from vidar.arch.networks.MVGD.AutoEncoderKL import AutoEncoderKL

from vidar.geometry.pose_utils import invert_pose

from vidar.utils.data import make_list, remove_nones_dict, get_from_dict
from vidar.utils.types import is_dict, is_tensor
from vidar.utils.config import Config
from vidar.utils.rays import Rays, intersect_skew_lines_high_dim
from vidar.utils.read import read_pickle


def prep_minmax(data, cfg):
    min_depth, max_depth = cfg.range
    if len(data.shape) == 4:
        ones = torch.ones(data.shape[0], 1, 1, 1, dtype=data.dtype, device=data.device)
    elif len(data.shape) == 3:
        ones = torch.ones(data.shape[0], 1, 1, dtype=data.dtype, device=data.device)
    else:
        raise ValueError('Invalid latent dimension')
    min_depth =  ones * min_depth
    max_depth =  ones * max_depth
    return min_depth, max_depth


def to_plucker(latent):
    b, c, h, w = latent.shape
    rays = Rays(
        rays=latent.view(b, c, -1).permute(0, 2, 1),
        is_plucker=False,
    )

    rays = rays.to_plucker()
    moments = rays.get_moments()
    directions = rays.get_directions()

    latent = torch.cat([directions, moments], -1)
    latent = latent.permute(0, 2, 1).view(b, c, h, w)
    return latent

def from_plucker(latent):

    if len(latent.shape) == 4:
        b, c, h, w = latent.shape
    elif len(latent.shape) == 3:
        b, c, n = latent.shape
    else:
        raise ValueError('Invalid latent shape')

    rays = Rays(
        rays=latent.view(b, c, -1).permute(0, 2, 1),
        is_plucker=True,
    )

    rays = rays.to_point_direction(normalize_moment=False)
    origins = rays.get_origins()
    directions = rays.get_directions()

    origins, _ = intersect_skew_lines_high_dim(origins, directions)
    if len(latent.shape) == 4:
        origins = origins.view(b, 3, 1, 1).repeat(1, 1, h, w)
        directions = directions.permute(0, 2, 1).reshape(b, 3, h, w)
    elif len(latent.shape) == 3:
        origins = origins.view(b, 3, 1).repeat(1, 1, n)
        directions = directions.permute(0, 2, 1)
    else:
        raise ValueError('Invalid latent shape')

    latent = torch.cat([origins, directions], 1)
    return latent

####################################################

def vq_4_kl(path):
    return Config(**{
        'checkpoint': path,
        'embed_dim': 3,
        'z_channels': 3,
        'double_z': True,
        'encoder': Config(**{
            'resolution': 256,
            'in_channels': 3,
            'ch': 128,
            'ch_mult': [1,2,4],
            'num_res_blocks': 2,
            'attn_resolutions': [],
            'dropout': 0.0,
        }),
        'decoder': Config(**{
            'resolution': 256,
            'out_ch': 3,
            'ch': 128,
            'ch_mult': [1,2,4],
            'num_res_blocks': 2,
            'attn_resolutions': [],
            'dropout': 0.0,
        })
    })

def prepare_latent(cfg, task):
    with_latent = cfg.has('tasks') and \
                  cfg.tasks.has(task) and \
                  cfg.tasks.dict[task].mode.startswith('latent')
    if with_latent:
        path = cfg.tasks.dict[task].mode.split('-')[-1]
        if os.environ['SAGEMAKER'] == 'enabled':
            root = '/opt/ml/input/data/training/models/rin'
        else:
            root = '/data/models/rin'
        cfg_latent = vq_4_kl(f'{root}/{path}.ckpt')
        cfg = Config(**{**cfg.dict, **cfg_latent})
    return with_latent, cfg

class AutoEncoderGeometryKL(AutoEncoderKL):

    def __init__(self, cfg):
        cfg = self.prepare_cfg(cfg)
        super().__init__(cfg)
        self.tasks = list(cfg.tasks.keys())
        self.cfg_tasks = cfg.tasks

        rgb_length = 4 if self.with_rgb_vae else 3
        depth_length = 4 if self.with_depth_vae else 1

        self.num_actions = 16
        self.dim_actions = 20

        self.lengths = {
            'rgb': rgb_length,
            'depth': depth_length,
            'points': 3,
            'camera': 6,
        }

        self.scale_factor_rgb = 0.05

        if self.with_rgb_vae:
            self.scale_factor_vae = 0.12135 # 320
            import torchvision
            self.transform = torchvision.transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            self.all_latents = []
            from diffusers.models import AutoencoderKL
            self.vae = AutoencoderKL.from_pretrained(f"madebyollin/sdxl-vae-fp16-fix")
            self.vae.eval()

        if cfg.has('tasks') and cfg.tasks.has('action') and cfg.tasks.action.has('normalizers'):
            self.normalizer = prep_normalizer(cfg)
        else:
            self.normalizer = None

    def prepare_cfg(self, cfg):
        self.with_rgb_latent, cfg = prepare_latent(cfg, 'rgb')
        self.with_video_latent, cfg = prepare_latent(cfg, 'video')
        self.with_rgb_vae = cfg.has('tasks') and \
                            cfg.tasks.has('rgb') and \
                            cfg.tasks.dict['rgb'].mode.startswith('vae')
        self.with_depth_vae = cfg.has('tasks') and \
                              cfg.tasks.has('depth') and \
                              cfg.tasks.dict['depth'].has('with_vae', False)
        return cfg

####################################################
        
    def encode_rgb(self, cfg, rgb):
        if cfg.mode.startswith('latent'):
            with torch.no_grad():

                rgb_norm = rgb
                latent = self.encode(rgb_norm).sample() * self.scale_factor_rgb

        elif cfg.mode.startswith('vae'):
            with torch.no_grad():
                rgb_norm = self.transform(rgb.clone())
                latent = self.vae.encode(rgb_norm).latent_dist.sample().mul_(self.scale_factor_vae)

        elif cfg.mode == 'normalize':
            latent = 2.0 * rgb - 1.0
        else:
            raise ValueError('Invalid encode rgb')
        return latent
    
    def encode_depth(self, cfg, rgb, depth, scales=None):
        min_depth, max_depth = cfg.range
        if scales is not None:
            min_depth = scales * min_depth
            max_depth = scales * max_depth
        else:
            ones = torch.ones(depth.shape[0], 1, 1, 1, dtype=depth.dtype, device=depth.device)
            min_depth =  ones * min_depth
            max_depth =  ones * max_depth
        if cfg.mode == 'linear':
            latent = depth / max_depth
            latent = 2.0 * latent - 1.0
        elif cfg.mode == 'normalize':
                min_ = torch.quantile(depth[depth != 0].to(torch.float), 0.02)
                max_ = torch.quantile(depth[depth != 0].to(torch.float), 0.98)
                latent = depth.clip(min_, max_)
                dmin, dmax =  latent.min(), latent.max()
                latent = (latent - dmin) / (dmax - dmin)
                latent = latent.clip(0.0, 1.0)
                latent = 2.0 * latent - 1.0
        elif cfg.mode == 'log-scale':
            latent = depth.clamp(min_depth, max_depth)
            latent = torch.log(latent / min_depth) / torch.log(max_depth / min_depth)
            latent = 2.0 * latent - 1.0
        return latent
    
    def encode_points(self, cfg, points):
        if cfg.mode == 'nothing':
            latent = points
        elif cfg.mode == 'normalize':
            latent = points
            latent[:, 2] = 2.0 * latent[:, 2] - 1.0
        elif cfg.mode == 'log-scale':

            min_depth, max_depth = prep_minmax(points, cfg)
            half_max_depth = max_depth / 2

            latent = points.clone()

            latent[:, [0]] = latent[:, [0]].clamp(- half_max_depth, half_max_depth) + half_max_depth + 1e-6
            latent[:, [1]] = latent[:, [1]].clamp(- half_max_depth, half_max_depth) + half_max_depth + 1e-6
            latent[:, [2]] = latent[:, [2]].clamp(min_depth, max_depth)

            latent = torch.log(latent / min_depth) / torch.log(max_depth / min_depth)
            latent = 2.0 * latent - 1.0

        else:
            raise ValueError('Invalid encode points')
        return latent

    def encode_camera(self, cfg, cams):
        orig = cams.float().scaled(cfg.downsample).get_origin(flatten=False)
        if cfg.has('global_rays', False):
            rays = cams.float().scaled(cfg.downsample).get_viewdirs(
                normalize=True, flatten=False, to_world=True)
        else:
            rays = cams.float().scaled(cfg.downsample).no_translation().get_viewdirs(
                normalize=True, flatten=False, to_world=True)
        latent = torch.cat([orig, rays], 1)
        if cfg.has('plucker', False):
            latent = to_plucker(latent)
        return latent

####################################################

    def decode_rgb(self, cfg, latent, shape=None):
        if cfg.mode.startswith('latent'):
            latent = latent.view(shape)
            with torch.no_grad():
                rgb = self.decode(latent / self.scale_factor_rgb)
        elif cfg.mode.startswith('vae'):
            latent = latent.view(shape)
            with torch.no_grad():
                rgb = self.vae.decode(latent / self.scale_factor_vae).sample
                rgb = (rgb + 1) / 2
        elif cfg.mode == 'normalize':
            rgb = (latent + 1.0) / 2.0
        else:
            raise ValueError('Invalid decode rgb')
        # if not self.training:
        rgb = torch.clip(rgb, min=0.0, max=1.0)
        return rgb

    def decode_depth(self, cfg, latent, scales=None):
        min_depth, max_depth = cfg.range
        if scales is not None:
            min_depth = scales * min_depth
            max_depth = scales * max_depth
        else:
            ones = torch.ones(latent.shape[0], 1, 1, dtype=latent.dtype, device=latent.device)
            min_depth =  ones * min_depth
            max_depth =  ones * max_depth
        if cfg.mode == 'linear':
            depth = (latent + 1.0) / 2.0
            depth = depth * max_depth
        elif cfg.mode == 'normalize':
            depth = (latent + 1.0) / 2.0
        elif cfg.mode == 'log-scale':
            depth = latent.clamp(min=-1.0, max=1.0)
            depth = (depth + 1.0) / 2.0
            depth = torch.exp(depth * torch.log(max_depth / min_depth)) * min_depth
            if scales is not None:
                depth = depth / scales
        else:
            raise ValueError('Invalid decode depth')
        if not self.training:
            depth = torch.clip(depth, min=cfg.range[0], max=cfg.range[1])
        return depth

    def decode_points(self, cfg, latent):
        if cfg.mode == 'nothing':
            points = latent
        elif cfg.mode == 'normalize':
            points = latent
            points[:, 2] = (points[:, 2] + 1.0) / 2.0
        elif cfg.mode == 'log-scale':
            min_depth, max_depth = prep_minmax(latent, cfg)
            half_max_depth = max_depth / 2
            points = latent.clamp(min=-1.0, max=1.0)
            points = (points + 1.0) / 2.0
            points = torch.exp(points * torch.log(max_depth / min_depth)) * min_depth
            points[:, [0]] = points[:, [0]] - half_max_depth
            points[:, [1]] = points[:, [1]] - half_max_depth
        return points

    def decode_camera(self, cfg, latent):
        orig, rays = latent[:, :3], latent[:, 3:]
        rays = rays / torch.norm(rays, dim=1).unsqueeze(1)
        camera = torch.cat([orig, rays], 1)
        if cfg.has('plucker', False):
            camera = from_plucker(camera)
        return camera

####################################################
    
    def to_latents(self, rgb=None, depth=None, action=None,
                   points=None, scnflow=None, cams=None,
                   keys=None, key=None, scales=None, tasks=None):
        tasks = make_list(tasks) if tasks is not None else self.tasks
        if keys is not None:
            return {
                key: self.to_latents(
                    rgb=get_from_dict(rgb, key),
                    depth=get_from_dict(depth, key),
                    action=get_from_dict(action, key),
                    scnflow=get_from_dict(scnflow, key),
                    cams=get_from_dict(cams, key),
                    key=key, scales=scales, tasks=tasks,
                ) for key in keys
            }
        data = []
        for task in tasks:
            cfg = self.cfg_tasks[task]
            if task == 'rgb':
                latent = self.encode_rgb(cfg, rgb)
            elif task == 'depth':
                latent = self.encode_depth(cfg, rgb, depth, scales=scales)
            elif task == 'points':
                points = cams.float().reconstruct_depth_map(depth, to_world=True, euclidean=True)
                latent = self.encode_points(cfg, points)
            elif task == 'camera':
                latent = self.encode_camera(cfg, cams)
            data.append(latent)
        return torch.cat(data, 1)

    def from_latents(self, latent, scales=None, cams=None, action_base=None, shape=None, tasks=None):
        tasks = make_list(tasks) if tasks is not None else self.tasks
        latent = latent.permute(0, 2, 1)
        idx, output = 0, {}
        for task in tasks:
            cfg, add = self.cfg_tasks[task], self.lengths[task]
            if task == 'rgb':
                output[task] = self.decode_rgb(cfg, latent[:, idx:idx+add], shape=shape)
            elif task == 'depth':
                output[task] = self.decode_depth(cfg, latent[:, idx:idx+add], scales=scales)
            elif task == 'points':
                output[task] = self.decode_points(cfg, latent[:, idx:idx+add])
                orig = cams[(0,0)].scaled(1.0).get_origin(flatten=True).permute(0, 2, 1)
                rays = output[task] - orig
                output['depth_points'] = (rays ** 2).sum(1, keepdim=True).sqrt()
                rays = output[task] / torch.norm(output[task] + 1e-6, dim=1).unsqueeze(1)
                output['camera_points'] = torch.cat([orig, rays], 1)
            elif task == 'camera':
                output[task] = self.decode_camera(cfg, latent[:, idx:idx+add])
            idx += add
        return output

    def decode_from_latents(self, latents):
        if is_dict(latents):
            return {key: self.decode_from_latents(val) for key, val in latents.items()}
        decoded = make_list(self.decode(latents))
        decoded = [self.from_latents(val) for val in decoded]

        return remove_nones_dict({
            'rgb': [dec['rgb'][0] for dec in decoded] if 'rgb' in decoded[0] else None,
            'depth': [dec['depth'][0] for dec in decoded] if 'depth' in decoded[0] else None,
            'camera': [dec['camera'][0] for dec in decoded] if 'camera' in decoded[0] else None,
        })

    def forward(self, rgb=None, depth=None, cams=None, sample_posterior=True):
        data = self.make_gt(rgb, depth, cams)
        _, decoded, _ = super().forward(data, sample_posterior)
        output = self.from_latents(decoded)
        return output

####################################################
