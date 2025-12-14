import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import math
import numpy as np

from model.autoencoder.autoencoder import AutoencoderKL
from model.HFDM import HLFAtt
from model.MoSE.MoSE_Net import HyperspectralRegionMoE
from model.score_model.unet import UNetModel


class NoiseScheduler:

    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x_start, t, noise=None, noise_scale=1.0):
        if noise is None:
            noise = torch.randn_like(x_start) * noise_scale

        device = x_start.device


        t = torch.clamp(t, 0, self.num_timesteps - 1)

        alphas_cumprod = self.alphas_cumprod.to(device)
        sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - alphas_cumprod[t]).view(-1, 1, 1, 1)

        x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return x_noisy, noise

    def get_alpha_cumprod(self, t, device):
        t_cpu = t.cpu()
        return self.alphas_cumprod[t_cpu].to(device)


class DDIMSampler:

    def __init__(self, scheduler, eta=0.0):
        self.scheduler = scheduler
        self.eta = eta

    @torch.no_grad()
    def sample(self, model, cond, shape, timesteps=50, return_intermediates=False):
        batch_size, channels, height, width = shape
        device = next(model.parameters()).device

        step = self.scheduler.num_timesteps // timesteps
        timestep_list = list(range(0, self.scheduler.num_timesteps, step))[::-1]

        x = torch.randn(shape, device=device)

        intermediates = []

        for i in range(len(timestep_list) - 1):
            t = torch.full((batch_size,), timestep_list[i], device=device, dtype=torch.long)
            t_next = torch.full((batch_size,), timestep_list[i + 1], device=device, dtype=torch.long)

            noise_pred = model(x, t, low_res_context=cond)


            alpha_t = self.scheduler.get_alpha_cumprod(t, device).view(-1, 1, 1, 1)
            alpha_next = self.scheduler.get_alpha_cumprod(t_next, device).view(-1, 1, 1, 1)

            x0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / (torch.sqrt(alpha_t) + 1e-8)
            x = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1 - alpha_next) * noise_pred

            if return_intermediates:
                intermediates.append(x.detach())

        if return_intermediates:
            return x, intermediates
        return x

    def deterministic_sample(self, model, cond, x_start, t):
        scheduler = self.scheduler

        device = x_start.device

        alphas_cumprod_t = scheduler.get_alpha_cumprod(t, device)
        sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod_t).view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - alphas_cumprod_t).view(-1, 1, 1, 1)

        noise = torch.randn_like(x_start)
        x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

        noise_pred = model(x_noisy, t, context=cond)

        x0_pred = (x_noisy - sqrt_one_minus_alphas_cumprod_t * noise_pred) / (sqrt_alphas_cumprod_t + 1e-8)

        return x0_pred


class MultiSpectralBranch(nn.Module):

    def __init__(self, in_channels=4, dims=[32, 32, 32]):
        super().__init__()

        self.proj = nn.Conv2d(in_channels, dims[0], kernel_size=1)
        self.hfdm1 = HLFAtt(dim=dims[0])
        self.hfdm2 = HLFAtt(dim=dims[1])
        self.hfdm3 = HLFAtt(dim=dims[2])

        self.adapters = nn.ModuleList([
            nn.Conv2d(in_channels, dims[0], kernel_size=1),
            nn.Conv2d(in_channels, dims[1], kernel_size=1),
            nn.Conv2d(in_channels, dims[2], kernel_size=1)
        ])

    def forward(self, x):
        x_init = self.proj(x)
        mf1 = self.hfdm1(x_init)
        mf2 = self.hfdm2(mf1)
        mf3 = self.hfdm3(mf2)
        return mf3


class ResidualBlock(nn.Module):
    def __init__(self, in_channels=64, hidden_channels=64, out_channels=103):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = self.relu3(self.conv3(out))
        res = out + residual

        res = self.relu4(self.conv4(res))
        return res


class ReconstructionModule(nn.Module):

    def __init__(self, in_channels=64, hidden_channels=64, out_channels=103):
        super().__init__()
        self.block1 = ResidualBlock(in_channels, hidden_channels, hidden_channels)
        self.block2 = ResidualBlock(in_channels, hidden_channels, hidden_channels)
        self.block3 = ResidualBlock(in_channels, hidden_channels, out_channels)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class MoSE(nn.Module):

    def __init__(self, partition_indices, in_channels=103, hidden_size=32,
                 num_regions=3, experts_per_region=3, top_k=2, projection_dim=64,
                 ms_channels=[32, 32, 32]):
        super().__init__()

        self.mose = HyperspectralRegionMoE(
            partition_indices=partition_indices,
            in_channels=in_channels,
            hidden_size=hidden_size,
            num_regions=num_regions,
            experts_per_region=experts_per_region,
            top_k=top_k,
            projection_dim=projection_dim
        )

    def forward(self, x, ms_features):
        output, aux_loss = self.mose(x, ms_features)

        return output, aux_loss


class FusionModel(nn.Module):

    def __init__(self, vae_config, vae_checkpoint_path,
                 partition_indices=[0, 23, 75, 103],
                 hs_channels=None,
                 ms_channels=4):
        super().__init__()

        self.scheduler = NoiseScheduler(num_timesteps=1000)
        self.ddim_sampler = DDIMSampler(self.scheduler)

        self.grad_clip_value = 1.0
        self.diff_loss_clip = 10.0

        self.vae = AutoencoderKL(vae_config, embed_dim=64)

        if vae_checkpoint_path:
            checkpoint = torch.load(vae_checkpoint_path, map_location='cpu')
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    vae_state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    vae_state_dict = checkpoint['state_dict']
                else:
                    vae_state_dict = checkpoint
            else:
                vae_state_dict = checkpoint

            self.vae.load_state_dict(vae_state_dict, strict=False)
            print("Loaded VAE weights from checkpoint")

        for name, param in self.vae.named_parameters():
            if 'encoder' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        print("VAE encoder frozen, decoder trainable")

        unet_config = {
            "image_size": 64,
            "in_channels": 64,
            "model_channels": 160,
            "out_channels": 64,
            "num_res_blocks": 2,
            "attention_resolutions": [8, 16],
            "dropout": 0.1,
            "channel_mult": [1, 2, 4, 4],
            "num_heads": 8,
            "num_head_channels": 32,
            "use_scale_shift_norm": True,
            "low_res_context_channels": 64
        }
        self.diffusion_model = UNetModel(**unet_config)

        self._init_diffusion_weights()

        self.ms_branch = MultiSpectralBranch(
            in_channels=ms_channels,
            dims=[32, 32, 32]
        )

        self.mose_net = MoSE(
            partition_indices=partition_indices,
            in_channels=hs_channels,
            hidden_size=32,
            num_regions=3,
            experts_per_region=3,
            top_k=2,
            projection_dim=64,
            ms_channels=[32, 32, 32]
        )

        self.reconstruction = ReconstructionModule(
            in_channels=64,
            hidden_channels=64,
            out_channels=hs_channels
        )

    def _init_diffusion_weights(self):
        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.diffusion_model.apply(init_weights)
        print("Diffusion model weights initialized")

    def normalize_latent(self, z):
        
        return (z - z.mean()) / (z.std() + 1e-8)

    def forward(self, lr_hs, ms_img, hr_hs=None, mode='train'):

        if mode == 'train' and hr_hs is None:
            raise ValueError("hr_hs must be provided in training mode")

        batch_size = lr_hs.shape[0]
        device = lr_hs.device


        with torch.no_grad():
            posterior_lr = self.vae.encode(lr_hs)
            z_cond = posterior_lr.sample()

            if mode == 'train':
                posterior_hr = self.vae.encode(hr_hs)
                z_target = posterior_hr.sample()


        z_cond = self.normalize_latent(z_cond)
        if mode == 'train':
            z_target = self.normalize_latent(z_target)

        if mode == 'train':

            t_max = self.scheduler.num_timesteps - 1
            t = torch.randint(0, t_max, (batch_size,), device=device).long()


            noise_scale = 1.0
            noise = torch.randn_like(z_target) * noise_scale

            z_noisy, noise = self.scheduler.add_noise(z_target, t, noise, noise_scale)


            noise_pred = self.diffusion_model(z_noisy, t, low_res_context=z_cond)

            alpha_t = self.scheduler.get_alpha_cumprod(t, device)

            diffusion_loss = F.mse_loss(noise_pred , noise , reduction='mean')


            sqrt_alphas_cumprod_t = torch.sqrt(alpha_t).view(-1, 1, 1, 1)
            sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - alpha_t).view(-1, 1, 1, 1)

            z_pred = (z_noisy - sqrt_one_minus_alphas_cumprod_t * noise_pred) / (sqrt_alphas_cumprod_t + 1e-8)
            latent_fused = z_pred.detach()

        else:

            with torch.no_grad():
                latent_shape = (batch_size, 64, 8, 8)
                z_pred = self.ddim_sampler.sample(self.diffusion_model, z_cond, latent_shape, timesteps=50)
                latent_fused = z_pred

        latent_fused  = self.normalize_latent(latent_fused)

        recon_hs = self.vae.decode(z_pred)

        ms_features = self.ms_branch(ms_img)
        mose_output, aux_loss = self.mose_net(recon_hs, ms_features)


        fused_output = self.reconstruction(mose_output)

        if mode == 'train':
            recon_loss = F.l1_loss(fused_output, hr_hs)


            diff_weight = 1.0


            total_loss = diff_weight * diffusion_loss + recon_loss +  aux_loss

            loss_dict = {
                'diffusion_loss': diffusion_loss.item(),
                'recon_loss': recon_loss.item(),
                'aux_loss': aux_loss.item(),
                'total_loss': total_loss.item(),
                'diff_weight': diff_weight
            }

            return total_loss, loss_dict, fused_output
        else:
            return fused_output

    def clip_gradients(self):
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_value)


class FusionLoss(nn.Module):

    def __init__(self, diff_weight=0.1, recon_weight=1.0, aux_weight=0.01):
        super().__init__()
        self.diff_weight = diff_weight
        self.recon_weight = recon_weight
        self.aux_weight = aux_weight

    def forward(self, pred, target, diffusion_loss=0, aux_loss=0):
        recon_loss = F.l1_loss(pred, target)

        total_loss = (self.diff_weight * diffusion_loss +
                      self.recon_weight * recon_loss +
                      self.aux_weight * aux_loss)

        loss_dict = {
            'diffusion_loss': diffusion_loss.item() if torch.is_tensor(diffusion_loss) else diffusion_loss,
            'recon_loss': recon_loss.item(),
            'aux_loss': aux_loss.item() if torch.is_tensor(aux_loss) else aux_loss,
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict





