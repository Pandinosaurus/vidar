import torch
import torch.nn as nn
import numpy as np
import random

import lpips
import piq

from einops import repeat
from tqdm import tqdm

from rin_pytorch import RIN
from rin_pytorch.rin_pytorch import LayerNorm
from vidar.arch.networks.layers.diffusion.DiffusionNet import setup_network
from vidar.utils.types import is_tensor, is_dict, is_list
from vidar.utils.data import remove_nones_dict, first_key, extract_batch, first_value, tensor_like, get_from_dict
from vidar.utils.config import get_folder_name
from vidar.utils.setup import load_class
from vidar.utils.augmentations import fit_batch, augment_canonical_batch

from vidar.arch.models.generic.GenericModel_predictions import copy_gt_data

from einops import rearrange
from einops.layers.torch import Rearrange
from ema_pytorch import EMA

from copy import deepcopy
from vidar.geometry.camera_motion import slerp, uzumaki
from vidar.utils.write import write_image
from vidar.utils.viz import viz_depth

from vidar.utils.write import write_image


#####################################################################

def check_valid(data):
    return (data != 0) & (~ torch.isnan(data))


def valid_camera(batch):
    stack = get_from_dict(batch, 'cams')
    if stack is None:
        return None
    stack_K = torch.stack(list([s.K for s in stack.values()]), -1)
    stack_Twc = torch.stack(list([s.Twc.T for s in stack.values()]), -1)
    valid_K = check_valid(stack_K[:, :3, :3].abs().sum([1,2,3]))
    valid_Twc = check_valid(stack_Twc.abs().sum([1,2,3]))
    return valid_K & valid_Twc


def valid_action(batch):
    stack = get_from_dict(batch, 'action')
    language = get_from_dict(batch, 'language')
    if stack is None or language is None:
        return None
    stack = torch.stack(list(stack.values()), -1)[:, 0]
    valid_action = check_valid(stack.abs().sum([1,2]))

    prompt = first_value(batch['language'])['prompt']
    valid_language = [l != '' for l in prompt]
    valid_language = torch.tensor(valid_language, device=valid_action.device)

    return valid_action & valid_language


def valid_video(batch):
    stack = get_from_dict(batch, 'time_cam')
    if stack is None:
        return None
    stack = torch.stack(list(stack.values()), -1)
    stack_time, stack_cam = stack[:, 0], stack[:, 1]
    valid_time = stack_time.abs().sum(1) != 0
    valid_cam = (stack_cam[:, [0]] - stack_cam).abs().sum(1) == 0
    return valid_cam & valid_time


def get_train_task(self, batch):
    if self.rin.task_embeddings is not None:
        tasks = list(self.rin.task_embeddings.keys())
        invalids = []
        while len(invalids) < len(tasks):
            tasks = [t for t in tasks if t not in invalids]
            task = random.sample(tasks, 1)[0]
            if task in ['rgb','depth']:
                valid = valid_camera(batch)
                if any(valid):
                    return task, None if all(valid) else valid
                else:
                    invalids.append(task)
            if task in ['action']:
                valid = valid_action(batch)
                if any(valid):
                    return task, None if all(valid) else valid
                else:
                    invalids.append('action')
            elif task in ['video']:
                valid = valid_video(batch)
                if any(valid):
                    return task, None if all(valid) else valid
                else:
                    invalids.append('video')
            else:
                return task, None
    else:
        return None, None


def check_val_task(task, batch):
    if task == 'action':
        valid = valid_action(batch)
        if valid is None or not all(valid):
            return False
    if task == 'video':
        valid = valid_video(batch)
        if valid is None or not all(valid):
            return False
    return True

#####################################################################

def revert_velocity(self, velocity, noise, timesteps):

    device, dtype, shape = velocity.device, velocity.dtype, velocity.shape

    self.alphas_cumprod = self.alphas_cumprod.to(device=device)
    alphas_cumprod = self.alphas_cumprod.to(dtype=dtype)
    timesteps = timesteps.to(device)

    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    sample = (sqrt_alpha_prod * noise - velocity) / sqrt_one_minus_alpha_prod
    return sample


def mean_flat(tensor, invalid=None):
    if invalid is None:
        return tensor.mean(dim=list(range(1, len(tensor.shape))))
    else:
        losses = []
        for b in range(tensor.shape[0]):
            losses.append(tensor[b][~invalid[b]].mean())
        return torch.stack(losses)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    if is_tensor(arr):
        res = arr[timesteps].float()
    else:
        res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def create_data(batch, cfg, keys):
    tag = get_from_dict(batch, 'tag')
    timestep = get_from_dict(batch, 'timestep')
    time_cam = get_from_dict(batch, 'time_cam')
    cams = get_from_dict(batch, 'cams')
    action = get_from_dict(batch, 'action')
    dummy_cam = first_value(cams).no_pose()
    return {key: {
        'cam': cams[key] if key in cams else dummy_cam,
        'cam_ctx': {
            (key[0] + n, key[1]): cams[(key[0] + n, key[1])]
                for n in [-1,1] if (key[0] + n, key[1]) in cams
        },
        'gt': {
            gt_key: copy_gt_data(batch, gt_key, key) for gt_key in cfg.gt_keys
        },
        'meta': {
            'action': get_from_dict(action, key),
            'timestep': get_from_dict(timestep, key),
            'time_cam': get_from_dict(time_cam, key),
            'tag': tag,
        }
    } for key in keys}

#####################################################################

class MVGD(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.lpips_loss = piq.LPIPS(reduction="none") # lpips.LPIPS(net='vgg')
        self.ssim_loss = piq.SSIMLoss(reduction="none", data_range=1.0) # lpips.LPIPS(net='vgg')

        self.flow_matching = cfg.has('flow_matching', False)
        self.depth_threshold = cfg.has('depth_threshold', 256)
        self.novel = cfg.has('novel', False)
        self.downsample = cfg.has('downsample', 1)

        self.deterministic = cfg.has('deterministic', False)
        self.remove_sparse = cfg.has('remove_sparse', True)
        self.points_mixed = cfg.has('points_mixed', False)

        if cfg.has('display') and cfg.display.has('enabled', True):
            from vidar.arch.models.generic.GenericModel_display import Display
            self.display = Display(cfg.display)
            self.display_mode = cfg.display.mode
        else:
            self.display = self.display_mode = None

        self.demo = cfg.has('demo', False)
        self.mixed_precision = cfg.has('mixed_precision', False)
        self.fit_batch = cfg.has('fit_batch', None)
        self.scale_depth_latents = cfg.has('scale_depth_latents', False)

        folder, name = get_folder_name(cfg.embeddings.ffile, 'networks')
        embeddings = load_class(name, folder)(cfg.embeddings)

        self.vq_model = None
        self.stages = torch.nn.ModuleDict()
        if cfg.has('stages'):
            for stage, stage_cfg in cfg.stages.dict.items():
                self.stages[stage] = setup_network(stage_cfg).eval()

        self.encode_scale_factor = 1.0
        self.decode_scale_factor = 1.0

        self.encode_sample = cfg.has('encode_sample', None)
        self.decode_sample = cfg.has('decode_sample', None)

        self.rin = RIN(cfg.rin, self.stages['geometry'].lengths)
        self.rin.embeddings = embeddings

        if cfg.has('packer'):
            self.rin.packer = Packer(cfg.packer)
        else:
            self.rin.packer = None

        if cfg.has('scheduler_rin'):
            from rin_pytorch import GDOI, ModelMeanType, ModelVarType, LossType
            self.scheduler_rin = GDOI(
                self.rin,
                model_mean_type=ModelMeanType.V,
                model_var_type=ModelVarType.FIXED_LARGE,
                loss_type=LossType.MSE,
                timesteps=cfg.scheduler_rin.timesteps,
                beta_schedule=cfg.scheduler_rin.beta_schedule,
            )
        else:
            self.scheduler_rin = None

        if cfg.has('scheduler_ddpm'):
            from rin_pytorch.DDPM import DDPMScheduler
            self.scheduler_ddpm = DDPMScheduler(**cfg.scheduler_ddpm.dict)
        elif cfg.has('scheduler_ddim'):
            from rin_pytorch.DDIM import DDIMScheduler
            self.scheduler_ddpm = DDIMScheduler(**cfg.scheduler_ddim.dict)
        else:
            self.scheduler_ddpm = None

        self.ema = EMA(
            self.rin, 
            beta=cfg.ema.beta,
            update_every=cfg.ema.update_every,
            update_after_step=cfg.ema.update_after_step,
        )

        self.train_prob_self_cond = 0.9
        self.mean_type = 'v-prediction'
        self.var_type = 'fixed_large'
        self._rescale_timesteps = False

        if self.scheduler_rin is not None:
            self.num_timesteps = self.scheduler_rin.num_timesteps
            snr = self.scheduler_rin.alphas_cumprod / (1 - self.scheduler_rin.alphas_cumprod)
        elif self.scheduler_ddpm is not None:
            self.num_timesteps = self.scheduler_ddpm.config.num_train_timesteps
            snr = self.scheduler_ddpm.alphas_cumprod / (1 - self.scheduler_ddpm.alphas_cumprod)
            snr = snr.cpu().numpy()
        else:
            raise ValueError('Invalid scheduler')
        maybe_clipped_snr = snr
        self.loss_weight = maybe_clipped_snr / (snr + 1)

        self.tasks_encode_idx = cfg.rin.has('tasks_encode_idx', None)
        self.tasks_decode_idx = cfg.rin.has('tasks_decode_idx', None)
        if self.tasks_encode_idx is not None:
            self.tasks_encode_idx = {key: val for key, val in zip(self.rin.tasks, self.tasks_encode_idx)}
        if self.tasks_decode_idx is not None:
            self.tasks_decode_idx = {key: val for key, val in zip(self.rin.tasks, self.tasks_decode_idx)}

        self.max_pool4 = torch.nn.MaxPool2d(4, stride=4)
        self.max_pool8 = torch.nn.MaxPool2d(8, stride=8)

        if cfg.rin.has('tasks_loss'):
            self.tasks_loss = {a: b for a, b in zip(cfg.rin.tasks, cfg.rin.tasks_loss)}
        else:
            self.tasks_loss = None

        if cfg.rin.has('tasks_weight'):
            self.tasks_weight = {a: b for a, b in zip(cfg.rin.tasks, cfg.rin.tasks_weight)}
        else:
            self.tasks_weight = None

        if cfg.has('filter_spikes'):
            from vidar.utils.logging import AvgMeter
            self.avg_losses = AvgMeter(100)
        else:
            self.avg_losses = None

        self.autoencoder = self.stages['geometry']
        self.with_rgb_latent = self.autoencoder.with_rgb_latent
        self.with_video_latent = self.autoencoder.with_video_latent
        self.with_rgb_vae = self.autoencoder.with_rgb_vae
        self.with_depth_vae = self.autoencoder.with_depth_vae
        self.incremental = cfg.has('incremental', None)
        self.augment = cfg.has('augment', None)

#####################################################################

    def get_downsample(self, task):
        if task == 'rgb' and self.with_rgb_latent:
            downsample = 4
        elif task == 'video' and self.with_video_latent:
            downsample = 4
        elif task == 'rgb' and self.with_rgb_vae:
            downsample = 8
        else:
            downsample = None
        return downsample

    def encode(self, stage, data):
        encoded = self.stages[stage].encode(data)
        sampled = encoded.sample() if not is_tensor(encoded) else encoded
        return self.encode_scale_factor * sampled 

    def decode(self, stage, latents):
        return self.stages[stage].decode(
            self.decode_scale_factor * latents)
    
    def extract_embeddings(self, embeddings, sample_mode, filename, rgb, shape=None, valid=None):
        if sample_mode is None:
            return None, embeddings, None
        coords, idx = {}, {}
        for key, val in embeddings.items():
            coords[key], idx[key], embeddings[key], _ = extract_batch(
                val, sample_mode, filename, rgb[key], shape, valid[key] if valid is not None else None)
        return coords, embeddings, idx

    def rescale_timesteps(self, t):
        if self._rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def dit_train(self, embeddings, noisy_x, timestep, language, task=None, idx=None):

        from vidar.utils.read import read_pickle

        device = noisy_x.device

        torch.manual_seed(self.args.seed)
        torch.set_grad_enabled(False)

        class_labels = [207, 360]

        n = len(class_labels)
        latent_size = self.args.image_size // 8
        z = torch.randn(n, 4, latent_size, latent_size, device=device)
        y = torch.tensor(class_labels, device=device)

        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * n, device=device)
        y = torch.cat([y, y_null], 0)

        encode_embeddings, decode_embeddings = embeddings
        encode_embeddings = torch.cat([val for val in encode_embeddings.values()], 1)
        decode_embeddings = torch.cat([val for val in decode_embeddings.values()], 1)
        embeddings = [encode_embeddings, decode_embeddings]

        model_kwargs = dict(
            y=y, 
            embeddings=embeddings,
        )

        b, n, c = noisy_x.shape
        s = int(n ** 0.5)

        z = noisy_x
        print('qwerqwer', z.shape, noisy_x.shape, decode_embeddings.shape)

        torch.manual_seed(self.args.seed)
        torch.set_grad_enabled(False)

        self.dit.eval()

        samples = self.diffusion.p_sample_loop(
            self.dit.forward, 
            z.shape, z, 
            clip_denoised=False, 
            model_kwargs=model_kwargs, 
            progress=True, 
            device=device,
        )

        samples = self.vae.decode(samples / 0.18215).sample

        print(samples.shape, samples.sum(), samples.mean())
        samples_gt = read_pickle('externals/fast_dit/data/dit/samples.pkl')
        print(samples_gt.shape, samples_gt.sum(), samples_gt.mean())

        import sys
        sys.exit()

    def rin_train(self, embeddings, noisy_x, timestep, language, task=None, idx=None):
        with torch.no_grad():
            self_latents = None
            _, self_latents = self.ema.online_model(
                noisy_x, timestep, embeddings,
                latent_self_cond=None, return_latents=True, 
                cond_drop_prob=0.0, task=task, idx=idx,
                language=language,
            )
            self_cond_mask = torch.rand((noisy_x.shape[0])) < self.train_prob_self_cond
            self_latents = torch.where(
                repeat(self_cond_mask.to(self_latents.device), 'b -> b n d',
                    n = self_latents.shape[1], d = self_latents.shape[2]),
                self_latents, 0)
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            output, invalid = self.ema.online_model(
                noisy_x, timestep, embeddings,
                latent_self_cond=self_latents, return_latents=False,
                cond_drop_prob=0.0, task=task, idx=idx,
                language=language,
            )

        return output, invalid

    def sample_timestep(self, latents):
        if not self.flow_matching:
            return torch.randint(0, self.num_timesteps, (latents.shape[0],), device=latents.device).long()
        else:
            timestamps = torch.rand((latents.shape[0],), device=latents.device)
            self.cont_t = timestamps.view(-1, 1, 1)
            return (timestamps * self.scheduler_ddpm.config.num_train_timesteps).long()

    def add_noise(self, x_start, timestep, noise):
        if not self.flow_matching:
            if self.scheduler_rin is not None:
                return self.scheduler_rin.q_sample(x_start, timestep, noise)
            elif self.scheduler_ddpm is not None:
                return self.scheduler_ddpm.add_noise(x_start, noise, timestep)
            else:
                raise ValueError('Invalid scheduler')
        else:
            x0, x1 = x_start, noise
            direction = x1 - x0
            return x0 + self.cont_t * direction

    def sample_noise(self, latents):
        return torch.randn_like(latents)

    def get_target(self, x_start, timestep, noise):
        if self.scheduler_rin is not None:
            return {
                'x_start': x_start, 'noise': noise,
                'v-prediction': self.scheduler_rin.predict_v(x_start, timestep, noise),
            }[self.mean_type]
        elif self.scheduler_ddpm is not None:
            return {
                'x_start': x_start, 'noise': noise,
                'v-prediction': self.scheduler_ddpm.get_velocity(x_start, noise, timestep),
            }[self.mean_type]
        else:
            raise ValueError('Invalid scheduler')

    def get_variance(self):
        return {
            'fixed_large': (
                np.append(self.scheduler_rin.posterior_variance[1], self.scheduler_rin.betas[1:]),
                np.log(np.append(self.scheduler_rin.posterior_variance[1], self.scheduler_rin.betas[1:])),
            ),
        }[self.var_type]

    def calc_loss(self, target, output, noise, timestep, tags, task=None, invalid=None):
        if task is None or self.tasks_loss is None:
            loss = (target - output) ** 2
        elif self.tasks_loss[task] == 'lat-l2':
            loss = (target - output) ** 2
        elif self.tasks_loss[task] == 'lat-l1':
            loss = (target - output).abs()
        elif self.tasks_loss[task].startswith('gt'):
            target = revert_velocity(self.scheduler_ddpm, target, noise, timestep)
            output = revert_velocity(self.scheduler_ddpm, output, noise, timestep)
            target = self.stages['geometry'].from_latents(target, tasks=task)[task].permute(0, 2, 1)
            output = self.stages['geometry'].from_latents(output, tasks=task)[task].permute(0, 2, 1)
            if self.tasks_loss[task] == 'gt-mse':
                loss = (target - output) ** 2
            elif self.tasks_loss[task] == 'gt-lpips':
                s = int(target[0].shape[0] ** 0.5)
                img0 = target.view(-1, s, s, 3).permute(0, 3, 1, 2)
                img1 = output.view(-1, s, s, 3).permute(0, 3, 1, 2)
                loss = self.lpips_loss(img0, img1)
            elif self.tasks_loss[task].startswith('gt-mse-lpips'):
                s = int(target[0].shape[0] ** 0.5)
                img0 = target.view(-1, s, s, 3).permute(0, 3, 1, 2)
                img1 = output.view(-1, s, s, 3).permute(0, 3, 1, 2)
                loss = (target - output) ** 2
                weight = float(self.tasks_loss[task].split('-')[-1])
                loss = loss + weight * self.lpips_loss(img0, img1).view(-1, 1, 1)
            elif self.tasks_loss[task] == 'gt-huber':
                loss = torch.nn.functional.huber_loss(output, target, reduction='none', delta=1.0)
            else:
                raise ValueError('Invalid diffusion loss')
        else:
            raise ValueError('Invalid diffusion loss')
        
        # Batch-wise loss       
        loss = mean_flat(loss, invalid)

        # Remove nan losses
        valid = ~ torch.isnan(loss)
        loss, timestep = loss[valid], timestep[valid]

        # Remove spiky losses
        if self.avg_losses is not None:
            if self.avg_losses.half_full():
                mean, std = self.avg_losses.mean(), self.avg_losses.std()
                valid = loss < (mean + 5.0 * std)
                if any(valid):
                    loss, timestep = loss[valid], timestep[valid]
                    self.avg_losses(loss.mean())
                else:
                    loss = loss * mean / loss.detach()
            else:
                self.avg_losses(loss.mean())

        timestep_weight = _extract_into_tensor(self.loss_weight, timestep, loss.shape)

        if task is not None and self.tasks_weight is not None:
            loss = loss * self.tasks_weight[task]

        return (loss * timestep_weight).mean()

    def training_loss(self, x_start, timestep, language, embeddings, noise, 
                      tags=None, task=None, idx=None):

        noisy_x = self.add_noise(x_start, timestep, noise)
        timestep = self.rescale_timesteps(timestep)

        if self.deterministic:
            noisy_x = torch.zeros_like(noisy_x)

        output, invalid = self.rin_train(embeddings, noisy_x, timestep, language, task=task, idx=idx)

        if not self.remove_sparse:
            invalid = x_start == -1

        target = self.get_target(x_start, timestep, noise)
        loss = self.calc_loss(target, output, noise, timestep, tags, task=task, invalid=invalid)

        return loss

    def forward_train(self, rgb=None, depth=None, cams=None, action=None, language=None,
                      mask_pad=None, scales=None, embeddings=None, filename=None,
                      tags=None, task=None):

        # Get embeddings data
        # encode_embeddings, decode_embeddings = embeddings
        task_encode_idx = self.tasks_encode_idx[task] if self.tasks_encode_idx is not None else 0
        task_decode_idx = self.tasks_decode_idx[task] if self.tasks_decode_idx is not None else 0
        encode_embeddings = {key: val for key, val in embeddings[0][task_encode_idx].items()}
        decode_embeddings = {key: val for key, val in embeddings[1][task_decode_idx].items()}
        embeddings_proc = [encode_embeddings, decode_embeddings]

        # Find valid decode embeddings
        if task in ['depth', 'points'] and self.remove_sparse:
            decode_valid = {key: depth[key] > 0.0 for key in decode_embeddings.keys()}
            floor = min([min([val[i].sum() for i in range(val.shape[0])]) for val in decode_valid.values()])
            if floor < self.depth_threshold:
                return self.forward_train(
                    rgb, depth, cams, action, language,
                    mask_pad, scales, embeddings, filename,
                    tags=tags, task='rgb')
        embeddings = embeddings_proc
        
        if task == 'camera':
            embeddings = [embeddings[1], {**embeddings[0], **embeddings[1]}]
            encode_embeddings = {key: val for key, val in embeddings[0].items()}
            decode_embeddings = {key: val for key, val in embeddings[1].items()}

        # Prepare pad masks
        if mask_pad is not None:
            encode_mask = {key: mask_pad[key] for key in encode_embeddings.keys()}
            decode_mask = {key: mask_pad[key] for key in decode_embeddings.keys()} if task not in ['action'] else None
        else:
            encode_mask = decode_mask = None

        # Find valid encode embeddings
        if encode_mask is not None:
            encode_valid = {}
            for key, val in encode_embeddings.items():
                down_mask = - self.max_pool4(- encode_mask[key])
                down_mask = down_mask.view(*down_mask.shape[:2], -1).permute(0, 2, 1) == 1.0
                encode_valid[key] = down_mask.squeeze(2).float()
        else:
            encode_valid = None

        # Find valid decode embeddings
        if task in ['depth','points'] and self.remove_sparse:
            decode_valid = {key: depth[key] > 0.0 for key in decode_embeddings.keys()}
            if decode_mask is not None:
                decode_valid = {key: decode_valid[key] * decode_mask[key] for key in decode_valid.keys()}
        else:
            decode_valid = decode_mask

        if decode_valid is not None:
            if (task == 'rgb' and self.with_rgb_vae) or (task == 'depth' and self.with_depth_vae):
                for key, val in decode_embeddings.items():
                    down_mask = - self.max_pool8(- decode_valid[key].float())
                    decode_valid[key] = down_mask.float()

        # Generate ground-truth data
        gt = self.stages['geometry'].to_latents(
            rgb=rgb, depth=depth, cams=cams, action=action,
            keys=decode_embeddings.keys(),
            scales=scales, tasks=task, 
        )

        # Subsample embeddings if requested
        shape = first_value(rgb).shape
        encode_sample = self.encode_sample if task in ['rgb','depth','points','video','action'] else None
        decode_sample = self.decode_sample if task in ['rgb','depth','points','video'] else None

        if is_list(encode_sample):
            encode_sample = encode_sample[self.rin.tasks.index(task)]
        if is_list(decode_sample):
            decode_sample = decode_sample[self.rin.tasks.index(task)]

        if task == 'rgb' and self.with_rgb_latent:
            decode_sample = None
        if task == 'video' and self.with_video_latent:
            decode_sample = None

        _, encode_embeddings, _ = self.extract_embeddings(
            encode_embeddings, encode_sample, filename, rgb, valid=encode_valid)
        _, decode_embeddings, decode_idx = self.extract_embeddings(
            decode_embeddings, decode_sample, filename, rgb, shape=shape, valid=decode_valid)

        # Subsample ground-truth if requested
        if decode_idx is not None:
            for key in gt.keys():
                gt[key] = gt[key].view(*gt[key].shape[:2], -1).permute(0, 2, 1)
                gt[key] = torch.stack([gt[key][i][decode_idx[key][i]] for i in range(gt[key].shape[0])], 0)
        else:
            for key in gt.keys():
                gt[key] = gt[key].view(*gt[key].shape[:2], -1).permute(0, 2, 1)
                gt[key] = torch.stack([gt[key][i] for i in range(gt[key].shape[0])], 0)

        # Encode ground-truth into latents
        # latents = {key: self.encode('geometry', val).detach() for key, val in gt.items()}
        latents = torch.cat([val.view(*val.shape[:2], -1) for val in gt.values()], 1)

        if self.rin.packer:

            tgt = (0,0)
            idx, valid = decode_idx[tgt], decode_valid[tgt]
            label = {'rgb': rgb, 'depth': depth}[task][tgt]
            self.ema.online_model.packer.prepare(task, label, idx, valid)

        # Sample timestep and noise
        timestep = self.sample_timestep(latents)
        noise = self.sample_noise(latents)

        # Calculate loss
        embeddings = [encode_embeddings, decode_embeddings]
        loss = self.training_loss(latents, timestep, language, embeddings, noise, 
                                  tags=tags, task=task, idx=task_encode_idx)

        # Update EMA
        self.ema.update()

        return {
            'predictions': {},
            'losses': {
                'diffusion': loss,
            },
        }

#####################################################################

    def extract_mean_variance(self, x_start, noisy_x, timestep):

        shape = noisy_x.shape

        posterior_mean = (
            _extract_into_tensor(self.scheduler_rin.posterior_mean_coef1, timestep, shape) * x_start + \
            _extract_into_tensor(self.scheduler_rin.posterior_mean_coef2, timestep, shape) * noisy_x
        )

        posterior_variance = _extract_into_tensor(
            self.scheduler_rin.posterior_variance, timestep, shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.scheduler_rin.posterior_log_variance_clipped, timestep, shape)
        
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def denoise_mean_variance(self, x, timestep, output, variance_noise=None):

        model_variance, model_log_variance = self.get_variance()
        model_variance = _extract_into_tensor(model_variance, timestep, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, timestep, x.shape)

        pred_xstart = self.scheduler_rin.predict_start_from_v(x, timestep, output)
        pred_xstart = pred_xstart.clamp(-1, 1)

        model_mean, _, _ = self.extract_mean_variance(
            x_start=pred_xstart, noisy_x=x, timestep=timestep,
        )

        if variance_noise is None:
            variance_noise = torch.randn_like(x)
        nonzero_mask = ((timestep != 0).float().view(-1, *([1] * (len(x.shape) - 1))))
        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * variance_noise

        return {
            "sample": sample,
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }
    
    def denoise_step(self, x, timestep, language, embeddings, last_latents=None, task=None, idx=None):
        
        variance_noise = torch.randn_like(x)

        output, latents = self.ema.ema_model(
            x, self.rescale_timesteps(timestep), embeddings, last_latents,
            return_latents=True, cond_drop_prob=0.0, task=task, idx=idx,
            language=language,
        )

        if self.scheduler_rin is not None:
            out = self.denoise_mean_variance(x, timestep, output,
                                             variance_noise=variance_noise)
            return {
                "sample": out["sample"],
                "pred_xstart": out["pred_xstart"],
                "last_latents": latents,
            }
        elif self.scheduler_ddpm is not None:
            if not self.flow_matching:
                out = self.scheduler_ddpm.step(output, timestep[[0]].cpu(), x,
                                               variance_noise=variance_noise)
                return {
                    "sample": out.prev_sample,
                    "pred_xstart": out.pred_original_sample,
                    "last_latents": latents,
                }
            else:
                prev_sample = x - output / self.scheduler_ddpm.config.num_inference_timesteps
                return {
                    "sample": prev_sample,
                    "last_latents": latents,
                }
        else:
            raise ValueError('Invalid scheduler')
    
    def denoise_rin(self, noise, embeddings, language=None, task=None, idx=None):
        x, last_latents = noise, None
        for i in list(range(self.num_timesteps))[::-1]:
            timestep = torch.tensor([i] * noise.shape[0], device=noise.device)
            with torch.no_grad():
                out = self.denoise_step(x, timestep, language, embeddings, last_latents, task=task, idx=idx)
                x, last_latents = out["sample"], out["last_latents"]
        return out['sample']

    def denoise_ddpm(self, noise, embeddings, language=None, task=None, idx=None):
        x, last_latents = noise, None
        if self.flow_matching:
            timesteps = torch.linspace(1, 0, self.scheduler_ddpm.config.num_inference_timesteps + 1, device=noise.device)[:-1]
            timesteps = (timesteps * self.scheduler_ddpm.config.num_train_timesteps).long()
            for timestep in self.scheduler_ddpm.timesteps:
                timestep = torch.tensor([timestep] * noise.shape[0], device=noise.device)
                with torch.no_grad():
                    out = self.denoise_step(x, timestep, language, embeddings, last_latents, task=task, idx=idx)
                    x, last_latents = out["sample"], out["last_latents"]
            return out['sample']
        else:
            self.scheduler_ddpm.set_timesteps(
                self.scheduler_ddpm.config.num_inference_timesteps,
                device=noise.device)
            for timestep in self.scheduler_ddpm.timesteps:
                timestep = torch.tensor([timestep] * noise.shape[0], device=noise.device)
                with torch.no_grad():
                    out = self.denoise_step(x, timestep, language, embeddings, last_latents, task=task, idx=idx)
                    x, last_latents = out["sample"], out["last_latents"]
            return out['sample']

    def forward_val(self, rgb=None, depth=None, cams=None, action=None, language=None, 
                    scales=None, embeddings=None, task=None):

        self.ema.ema_model.eval()
        with torch.no_grad():

            if self.rin.packer:
                tgt = (0,0)
                label = {'rgb': rgb, 'depth': depth}[task][tgt]
                self.ema.ema_model.packer.prepare(task, label)

            encode_idx = self.tasks_encode_idx[task] if self.tasks_encode_idx is not None else 0
            decode_idx = self.tasks_decode_idx[task] if self.tasks_decode_idx is not None else 0
            embeddings = [embeddings[0][encode_idx], embeddings[1][decode_idx]]

            if task == 'camera':
                embeddings = [embeddings[1], {**embeddings[0], **embeddings[1]}]

            decode_embeddings = {key: val for key, val in embeddings[1].items()}
            keys = decode_embeddings.keys()
            num_keys = len(decode_embeddings.keys())

            if task not in ['action']:
                shape = {key: rgb[key].shape for key in decode_embeddings.keys()}
                shape = first_value(shape)
                downsample = self.get_downsample(task)
                if downsample is None: downsample = 1
                shape_raw = [*shape[:2], shape[2] // downsample, shape[3] // downsample]
                shape_raw[1] = self.rin.task_channels(task)
                shape = (shape_raw[0], shape_raw[2] * shape_raw[3] * num_keys, shape_raw[1])
            else:
                shape = {key: action[key].shape for key in decode_embeddings.keys()}
                shape = first_value(shape)
                shape_raw = shape = [shape[0], num_keys, (shape[1] - 1) * shape[2]]
            noise = torch.randn(shape, device='cuda')

            if self.deterministic:
                noise = torch.zeros_like(noise)

            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                if self.scheduler_rin is not None:
                    latents_pred = self.denoise_rin(noise, embeddings, language=language, task=task)
                elif self.scheduler_ddpm is not None:
                    latents_pred = self.denoise_ddpm(noise, embeddings, language=language, task=task)
                else:
                    raise ValueError('Invalid scheduler')

            action_base = None if action is None else action[(0,0)][:, 0]
            output = self.stages['geometry'].from_latents(
                latents_pred, scales=scales, cams=cams, action_base=action_base, shape=shape_raw, tasks=task)

            for task in ['rgb', 'depth', 'camera', 'depth_points', 'camera_points', 'video']:
                if task in output.keys():
                    if len(output[task].shape) == 3:
                        output_task = list(torch.chunk(output[task], chunks=num_keys, dim=-1))
                        for i in range(len(output_task)):
                            output_task[i] = output_task[i].view(
                                output_task[i].shape[0], output_task[i].shape[1], shape_raw[2], shape_raw[3])
                        output[task] = {key: [val] for key, val in zip(keys, output_task)}
                    elif len(output[task].shape) == 4:
                        output_task = list(torch.chunk(output[task], chunks=num_keys, dim=-1))
                        output[task] = {key: [val] for key, val in zip(keys, output_task)}
                    else:
                        raise ValueError('Invalid output dimension')
            for task in ['action']:
                if task in output.keys():
                    output_task = list(torch.chunk(output[task], chunks=num_keys, dim=1))
                    output[task] = {key: [val] for key, val in zip(keys, output_task)}

            predictions = remove_nones_dict({
                key: output[key] if key in output else None for key in [
                    'rgb', 'depth', 'camera', 'action', 'depth_points', 'camera_points', 'video',
            ]})

        return {
            'losses': {},
            'predictions': predictions,
        }
    
#####################################################################

    def forward_incremental(self, batch, predictions, model, extra, epoch):

        filename = batch['filename'][(0,0)][0]
        filename = '/'.join(filename.split('/')[-5:-3])

        time_cam = batch['time_cam']
        time = {key: val[:, 0] for key, val in time_cam.items()}
        keys, vals = list(time.keys()), list(time.values())
        ordered = sorted(range(len(vals)), key=lambda k: vals)
        keys = [keys[i] for i in ordered]
        vals = [vals[i] for i in ordered]

        if self.incremental.has('range'):
            keys = keys[:self.incremental.range]

        n = len(keys)
        stride = self.incremental.stride

        enc = list(set([int(np.floor(v / stride) * stride) for v in list(range(n))]))
        encs = [keys[i] for i in enc]

        cams = batch['cams']
        enc_xyz = torch.stack([
            cams[key].Twc.T[:, :3, -1] for key in encs], 1).unsqueeze(-2)
        xyz = torch.stack([
            cams[key].Twc.T[:, :3, -1] for key in keys], 1).unsqueeze(1).repeat(1, enc_xyz.shape[1], 1, 1)

        incremental = self.incremental.incremental
        order = self.incremental.order
        if not incremental:
            order = 'time_cam'

        if order == 'distance':
            xyz = (xyz - enc_xyz).pow(2.0).sum(-1).min(1)[0]
            _, idx = torch.sort(xyz, dim=1, descending=False)
            keys = [keys[i] for i in idx[0]]

        all = {'pred': {}, 'cond': {}, 'gt': {}}
        all['cond'] = {key: {
            'rgb': batch['rgb'][key], 
            'depth': batch['depth'][key], 
            'cams': batch['cams'][key],
        } for key in encs}
        all['gt'] = {key: {
            'rgb': batch['rgb'][key], 
            'depth': batch['depth'][key], 
            'cams': batch['cams'][key],
        } for key in keys}

        predictions = {key1: {} for key1 in self.rin.tasks}
        progress = tqdm(keys) if self.incremental.has('progress', False) else keys
        for tgt in progress:
            added = [] if not incremental else list(first_value(predictions).keys())
            batch_filtered = {}
            for key1, val1 in batch.items():
                if is_dict(val1):
                    batch_filtered[key1] = {}
                    for key2, val2 in val1.items():
                        if key2 in encs + added + [tgt, (0,0)]:
                            batch_filtered[key1][key2] = val2 if is_list(val2) else val2.clone()
                else:
                    batch_filtered[key1] = val1
            if incremental:
                for key1, val1 in predictions.items():
                    for key2, val2 in predictions[key1].items():
                        if key2 not in encs:
                            batch_filtered[key1][key2] = val2[0].clone()

            batch_filtered = augment_canonical_batch(batch_filtered, base_key=tgt)

            encs_added = encs + added + [tgt]
            if tgt not in encs:
                encs_added = [v for v in encs_added if v != (0,0)]
            encs_added = list(set(encs_added))

            output =  self.forward(
                batch_filtered, predictions, model, extra, epoch, 
                novel=False, incremental=[encs_added, (0,0)])
            for key1, val1 in output['predictions'].items():
                predictions[key1][tgt] = output['predictions'][key1][(0,0)]


            all['pred'][tgt] = {
                **{key: val[tgt] for key, val in predictions.items()},
                'cams': batch_filtered['cams'][(0,0)]
            }

        return {
            'losses': {},
            'predictions': predictions,
        }        

#####################################################################

    def forward(self, batch, predictions, model, extra, epoch, novel=False, incremental=None):

        if self.training:
            if self.rin.task_embeddings is None:

                with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                    embeddings, decode_tags, batch = self.get_embeddings(batch, model)

                rgb = get_from_dict(batch, 'rgb')
                depth = get_from_dict(batch, 'depth')
                cams = get_from_dict(batch, 'cams')
                mask_pad = get_from_dict(batch, 'mask_pad')
                action = get_from_dict(batch, 'action')
                language = get_from_dict(batch, 'language') if task in ['action'] else None
                scales_forward = None

                output = self.forward_train(
                    rgb, depth, cams, action, language, mask_pad, scales_forward,
                    embeddings, tags=decode_tags, task=None)
                
            else:

                task, valid = get_train_task(self, batch)

                if valid is not None:
                    for key1, val1 in batch.items():
                        if is_dict(val1):
                            if key1 in ['language']:
                                batch[key1] = {key2: {key3: val3[valid] if is_tensor(val3) else [val3[i] for i, v in enumerate(valid) if v]
                                    for key3, val3 in val2.items()} for key2, val2 in batch[key1].items()}
                            else:
                                batch[key1] = {key: val[valid] 
                                    for key, val in batch[key1].items()}

                with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                    embeddings, decode_tags, batch = self.get_embeddings(batch, model, task=task)

                rgb = get_from_dict(batch, 'rgb')
                depth = get_from_dict(batch, 'depth')
                cams = get_from_dict(batch, 'cams')
                mask_pad = get_from_dict(batch, 'mask_pad')
                action = get_from_dict(batch, 'action')
                language = get_from_dict(batch, 'language') if task in ['action'] else None
                filename = get_from_dict(batch, 'filename')
                scales_forward = None

                output = self.forward_train(
                    rgb, depth, cams, action, language, mask_pad, scales_forward,
                    embeddings, filename, tags=decode_tags, task=task)

        else:

            predictions = {}
            if self.rin.tasks is not None:

                if self.incremental is not None and incremental is None:
                    return self.forward_incremental(batch, predictions, model, extra, epoch)
                elif incremental is not None:
                    encs, tgts = incremental[0], [incremental[1]]
                else:
                    encs = tgts = None

                if self.augment is not None:

                    tgt = (0,0)
                    for key1, val1 in batch.items():
                        if is_dict(val1):
                            val2 = batch[key1][tgt]
                            last = max([v[1] for v in val1.keys()]) + 1
                            batch[key1][(0,last)] = val2 if is_list(val2) else val2.clone()
                        else:
                            batch[key1] = val1
                    last = (0,last)

                    forward = batch['cams'][tgt].Twc.clone().translateForward(2.0).inverse().T[0, :3, -1].unsqueeze(0)
                    rnd1 = (random.random() - 1) * 2 / 10
                    rnd2 = (random.random() - 1) * 2 / 10

                    batch['cams'][last].Twc.translateUp(rnd1)
                    batch['cams'][last].Twc.translateLeft(rnd2)
                    batch['cams'][last].look_at(at=forward)

                    batch = augment_canonical_batch(batch, base_key=last)

                for task in self.rin.tasks:

                    if not check_val_task(task, batch):
                        continue

                    with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                        embeddings, decode_tags, batch_filtered = self.get_embeddings(
                            batch, model, task=task, encs=encs, tgts=tgts) 

                    rgb = get_from_dict(batch_filtered, 'rgb')
                    depth = get_from_dict(batch_filtered, 'depth')
                    cams = get_from_dict(batch_filtered, 'cams')
                    action = get_from_dict(batch_filtered, 'action')
                    mask_pad = get_from_dict(batch_filtered, 'mask_pad')
                    language = get_from_dict(batch_filtered, 'language') if task in ['action'] else None
                    scales_forward = None

                    if task == 'rgb':
                        output = self.forward_val(
                            rgb, depth, cams, action, language,
                            scales_forward, embeddings, task=task)
                        predictions['rgb'] = output['predictions']['rgb']
                    elif task == 'depth':
                        output = self.forward_val(
                            rgb, depth, cams, action, language,
                            scales_forward, embeddings, task=task)
                        predictions['depth'] = output['predictions']['depth']
                        scales = get_from_dict(batch_filtered, 'fit_scale')
                        if scales is not None:
                            for key in predictions['depth'].keys():
                                for i in range(len(predictions['depth'][key])):
                                    predictions['depth'][key][i] /= scales
                    elif task == 'points':
                        output = self.forward_val(
                            rgb, depth, cams, action, language, 
                            scales_forward, embeddings, task=task)
                        predictions['depth_points'] = output['predictions']['depth_points']
                        predictions['camera_points'] = output['predictions']['camera_points']
                        scales = get_from_dict(batch_filtered, 'fit_scale')
                        if scales is not None:
                            for key in predictions['depth_points'].keys():
                                for i in range(len(predictions['depth_points'][key])):
                                    predictions['depth_points'][key][i] /= scales
                                    predictions['camera_points'][key][i][:, :3] /= scales
                    elif task == 'camera':
                        output = self.forward_val(
                            rgb, depth, cams, action, language,
                            scales_forward, embeddings, task=task)
                        predictions['camera'] = output['predictions']['camera']
                        scales = get_from_dict(batch_filtered, 'fit_scale')
                        if scales is not None:
                            for key in predictions['camera'].keys():
                                for i in range(len(predictions['camera'][key])):
                                    predictions['camera'][key][i][:, :3] /= scales
                    else:
                        raise ValueError('Invalid tasks')

                output = {
                    'losses': {},
                    'predictions': predictions,
                }

            else:

                with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                    embeddings, decode_tags, batch_filtered = self.get_embeddings(batch, model, task=task)

                rgb = get_from_dict(batch_filtered, 'rgb')
                depth = get_from_dict(batch_filtered, 'depth')
                cams = get_from_dict(batch_filtered, 'cams')
                action = get_from_dict(batch_filtered, 'action')
                mask_pad = get_from_dict(batch_filtered, 'mask_pad')
                language = get_from_dict(batch_filtered, 'language') if task in ['action'] else None
                scales_forward = None

                output = self.forward_val(
                    rgb, depth, cams, action, language, 
                    scales_forward, embeddings)

            if self.demo:
                self.run_demo(batch_filtered, model, output)

        if self.display_mode == 'define':
            self.display.loop_define(batch_filtered, predictions, {}, None)

        return output

#####################################################################

    def get_embeddings(self, batch, model, task=None, idx=None, encs=None, tgts=None):

        dataset_prefix = get_from_dict(batch, 'dataset_prefix')
        network = self.ema.online_model.embeddings if self.training else self.ema.ema_model.embeddings
        cfg = model.task_cfg['embeddings']

        if encs is None and tgts is None:
            cams = batch['cams']
            encs = model.get_encodes(cfg, model.base_cam, cams.keys(), dataset_prefix=dataset_prefix)
            tgts = model.get_targets(cfg, model.base_cam, cams.keys(), dataset_prefix=dataset_prefix,
                forbidden=encs if network.remove_encs else [])
            if not network.decode_encodes:
                tgts = [tgt for tgt in tgts if tgt not in encs]

        downsample = self.get_downsample(task)

        if self.fit_batch is not None:
            keys = set(list(encs) + list(tgts))
            filtered_batch = {}
            for key1, val in batch.items(): 
                if is_dict(val):
                    filtered_batch[key1] = {key: val for key, val in val.items() if key in keys}
                else:
                    filtered_batch[key1] = val
            batch = filtered_batch
            batch = fit_batch(batch, self.fit_batch, tgts=tgts)

        encode_data = create_data(batch, cfg, encs)
        decode_data = create_data(batch, cfg, tgts)

        scene = get_from_dict(batch, 'scene')
        language = get_from_dict(batch, 'language')

        encoded_data = network.encode(
            data=encode_data, scene=scene, language=language,
        )

        decode_gt = network.get_gt_embeddings(decode_data)
        for key1 in decode_gt.keys():
            for key2 in decode_gt[key1].keys():
                decode_data[key2]['gt'][key1] = decode_gt[key1][key2]

        decode_output = network.decode(
            encoded=encoded_data, encode_data=encode_data, decode_data=decode_data,
            downsample=downsample,
        )

        decode_tags = get_from_dict(first_value(decode_data)['meta'], 'tag')

        encode_embeddings = [val['embeddings'] for val in encoded_data]
        decode_embeddings = [val for key, val in decode_output['embeddings'].items() if key.startswith('source')]
        embeddings = [encode_embeddings, decode_embeddings]

        return embeddings, decode_tags, batch

#####################################################################