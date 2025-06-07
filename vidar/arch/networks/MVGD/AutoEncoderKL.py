import torch
from vidar.arch.networks.layers.diffusion.DiagonalGaussianDistribution import DiagonalGaussianDistribution
from vidar.arch.networks.layers.diffusion.Encoder import Encoder
from vidar.arch.networks.layers.diffusion.Decoder import Decoder
from vidar.utils.data import load_pretrained, load_checkpoint, make_list
from vidar.utils.types import is_dict


class AutoEncoderKL(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = Encoder(cfg.encoder, z_channels=cfg.z_channels) if cfg.has('encoder') else None
        self.decoder = Decoder(cfg.decoder, z_channels=cfg.z_channels) if cfg.has('decoder') else None

        self.learn_logvar = cfg.has('learn_logvar', False)
        self.image_key = cfg.has('image_key', 'image')
        self.embed_dim = cfg.has('embed_dim', None)

        z_channels = cfg.has('z_channels', None)
        self.quant_conv = torch.nn.Conv2d(2 * z_channels, 2 * self.embed_dim, 1) if z_channels is not None else None
        self.post_quant_conv = torch.nn.Conv2d(cfg.embed_dim, z_channels, 1) if z_channels is not None else None

        self = load_pretrained(self, cfg, 'first_stage_model')
        self = load_checkpoint(self, cfg)

    def encode(self, x):
        if self.encoder is None:
            return x
        self.encoder.eval()
        encoded = self.encoder(x)
        moments = self.quant_conv(encoded)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        if self.decoder is None:
            return z
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return z, dec, posterior

    def decode_to_predictions(self, latents):
        if is_dict(latents):
            return {key: self.decode_to_predictions(val) for key, val in latents.items()}
        decoded = make_list(self.decode(latents))
        return {
            'rgb': decoded,
        }

    @staticmethod
    def get_input(batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface
        super().__init__()

    @staticmethod
    def encode(x, *args, **kwargs):
        return x

    @staticmethod
    def decode(x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    @staticmethod
    def forward(x, *args, **kwargs):
        return x

