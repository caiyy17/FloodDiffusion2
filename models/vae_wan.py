import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tools.wan_vae import WanVAE_
from .normalization import load_statistics
from .quadratic_loss import load_quadratic_loss_matrix, quadratic_error_sum


class VAEWanModel(nn.Module):
    def __init__(
        self,
        input_dim,
        mean_path=None,
        std_path=None,
        z_dim=256,
        dim=160,
        dec_dim=512,
        num_res_blocks=1,
        dropout=0.0,
        dim_mult=[1, 1, 1],
        temperal_downsample=[True, True],
        spatial_downsample=[False, False],
        spatial_dim=0,
        loss_W=None,
        loss_w_coefficient=1.0,
        input_keys={
            "feature": "feature",
            "feature_length": "feature_length",
        },
        **kwargs,
    ):
        super().__init__()
        self.input_keys = input_keys

        self.mean_path = mean_path
        self.std_path = std_path
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.dim = dim
        self.dec_dim = dec_dim
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.dim_mult = dim_mult
        self.temperal_downsample = temperal_downsample
        self.spatial_downsample = spatial_downsample
        self.spatial_dim = spatial_dim
        self.RECONS_LOSS = nn.SmoothL1Loss(reduction="none")
        self.register_buffer(
            "loss_matrix",
            load_quadratic_loss_matrix(
                loss_W, loss_w_coefficient, self.input_dim
            ),
        )
        self.LAMBDA_FEATURE = kwargs.get("LAMBDA_FEATURE", 1.0)
        self.LAMBDA_KL = kwargs.get("LAMBDA_KL", 10e-6)

        # Per-dimension reconstruction weights (default: all ones)
        # If shorter than input_dim, pad with 1s at the end.
        recons_weights = kwargs.get("recons_weights", None)
        if recons_weights is not None:
            w = torch.tensor(recons_weights, dtype=torch.float32)
            if w.numel() < input_dim:
                w = torch.cat([w, torch.ones(input_dim - w.numel())])
            self.register_buffer("recons_weights", w[:input_dim], persistent=False)
        else:
            self.register_buffer(
                "recons_weights", torch.ones(input_dim, dtype=torch.float32), persistent=False
            )

        mean, std = load_statistics(self.mean_path, self.std_path, input_dim)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        self.model = WanVAE_(
            input_dim=self.input_dim,
            dim=self.dim,
            dec_dim=self.dec_dim,
            z_dim=self.z_dim,
            dim_mult=self.dim_mult,
            num_res_blocks=self.num_res_blocks,
            temperal_downsample=self.temperal_downsample,
            spatial_downsample=self.spatial_downsample,
            spatial_dim=self.spatial_dim,
            dropout=self.dropout,
        )

        downsample_factor = 1
        for flag in self.temperal_downsample:
            if flag:
                downsample_factor *= 2
        self.downsample_factor = downsample_factor

    def _extract_inputs(self, x):
        inputs = {}
        for internal_key, external_key in self.input_keys.items():
            if external_key in x:
                inputs[internal_key] = x[external_key]
        return inputs

    def preprocess(self, x):
        """Convert last-channel batched format to channel-first, padding to 5D (B, C, T, H, W).
        spatial_dim=0: (B, T, C) -> (B, C, T, 1, 1)
        spatial_dim=1: (B, T, H, C) -> (B, C, T, H, 1)
        spatial_dim=2: (B, T, H, W, C) -> (B, C, T, H, W)
        """
        expected_ndim = self.spatial_dim + 3
        assert x.ndim == expected_ndim, (
            f"spatial_dim={self.spatial_dim} expects input ndim={expected_ndim}, "
            f"got ndim={x.ndim} with shape={tuple(x.shape)}"
        )
        if self.spatial_dim == 0:  # (B, T, C)
            x = x.permute(0, 2, 1)[:, :, :, None, None]
        elif self.spatial_dim == 1:  # (B, T, H, C)
            x = x.permute(0, 3, 1, 2)[:, :, :, :, None]
        else:  # spatial_dim == 2: (B, T, H, W, C)
            x = x.permute(0, 4, 1, 2, 3)
        return x

    def postprocess(self, x):
        """Reverse of preprocess: channel-first 5D back to last-channel, stripping padding dims.
        Uses self.spatial_dim (set at construction) instead of inferring from shape, so
        edge cases where H=1 in 2D data are not misclassified as 1D.
        spatial_dim=0: (B, C, T, 1, 1) -> (B, T, C)
        spatial_dim=1: (B, C, T, H, 1) -> (B, T, H, C)
        spatial_dim=2: (B, C, T, H, W) -> (B, T, H, W, C)
        """
        if self.spatial_dim == 0:
            x = x[:, :, :, 0, 0].permute(0, 2, 1)
        elif self.spatial_dim == 1:
            x = x[:, :, :, :, 0].permute(0, 2, 3, 1)
        else:  # spatial_dim == 2
            x = x.permute(0, 2, 3, 4, 1)
        return x

    def forward(self, x):
        x = self._extract_inputs(x)
        features = x["feature"]
        feature_length = x["feature_length"]
        features = (features - self.mean) / self.std
        # create mask based on feature_length
        batch_size, seq_len = features.shape[:2]
        mask = torch.zeros(
            batch_size, seq_len, dtype=torch.bool, device=features.device
        )
        for i in range(batch_size):
            mask[i, : feature_length[i]] = True

        x_in = self.preprocess(features)  # (bs, input_dim, T, 1, 1)
        mu, log_var = self.model.encode(
            x_in, scale=[0, 1], return_dist=True
        )  # (bs, z_dim, T, 1, 1)
        z = self.model.reparameterize(mu, log_var)
        x_decoder = self.model.decode(z, scale=[0, 1])  # (bs, input_dim, T, 1, 1)
        x_out = self.postprocess(x_decoder)  # (bs, T, input_dim)

        if x_out.size(1) != features.size(1):
            min_len = min(x_out.size(1), features.size(1))
            x_out = x_out[:, :min_len]
            features = features[:, :min_len]
            mask = mask[:, :min_len]

        mask_expanded = mask
        for _ in range(features.ndim - 2):
            mask_expanded = mask_expanded.unsqueeze(-1)
        # Weighted mean over (valid_time * spatial * channel). For 1D
        # (no spatial dim), spatial_numel == 1, matching the previous formula.
        spatial_numel = 1
        for s in features.shape[2:-1]:
            spatial_numel *= s
        if self.loss_matrix is None:
            # Preserve the original VAE objective exactly when loss_W is absent.
            loss_per_element = self.RECONS_LOSS(x_out, features)
            loss_recons = (
                loss_per_element * mask_expanded * self.recons_weights
            ).sum() / (
                mask_expanded.sum() * spatial_numel * self.recons_weights.sum()
            )
        else:
            error = (x_out - features) * mask_expanded
            loss_recons = quadratic_error_sum(
                error.movedim(-1, 0), self.loss_matrix
            ) / (mask.sum() * spatial_numel * self.input_dim)

        # Compute KL divergence loss
        # KL(N(mu, sigma) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # log_var = log(sigma^2), so we can use it directly

        # Build mask for latent space
        T_latent = mu.size(2)
        mask_downsampled = torch.zeros(
            batch_size, T_latent, dtype=torch.bool, device=features.device
        )
        for i in range(batch_size):
            latent_length = (
                feature_length[i] + self.downsample_factor - 1
            ) // self.downsample_factor
            mask_downsampled[i, :latent_length] = True
        mask_latent = (
            mask_downsampled.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        )  # (B, 1, T_latent, 1, 1)

        # Compute KL loss per element
        kl_per_element = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
        # Apply mask: only compute KL loss for valid timesteps
        kl_masked = kl_per_element * mask_latent
        # Sum over all dimensions and normalize by the number of valid elements
        num_latent_elements = mu.size(1) * mu.size(3) * mu.size(4)  # C * H * W
        kl_loss = torch.sum(kl_masked) / (
            torch.sum(mask_downsampled) * num_latent_elements
        )  # normalize by valid timesteps * (C * H * W)

        # Total loss
        total_loss = (
            self.LAMBDA_FEATURE * loss_recons
            + self.LAMBDA_KL * kl_loss
        )

        loss_dict = {}
        loss_dict["total"] = total_loss
        loss_dict["recons"] = loss_recons
        loss_dict["kl"] = kl_loss

        return loss_dict

    def encode(self, x):
        x = (x - self.mean) / self.std
        x_in = self.preprocess(x)  # (bs, T, input_dim) -> (bs, input_dim, T, 1, 1)
        mu = self.model.encode(x_in, scale=[0, 1])  # (bs, z_dim, T, 1, 1)
        mu = self.postprocess(mu)  # (bs, T, z_dim)
        return mu

    def decode(self, mu):
        mu_in = self.preprocess(mu)  # (bs, T, z_dim) -> (bs, z_dim, T, 1, 1)
        x_decoder = self.model.decode(mu_in, scale=[0, 1])  # (bs, z_dim, T, 1, 1)
        x_out = self.postprocess(x_decoder)  # (bs, T, input_dim)
        x_out = x_out * self.std + self.mean
        return x_out

    @torch.no_grad()
    def stream_encode(self, x, first_chunk=True):
        x = (x - self.mean) / self.std
        x_in = self.preprocess(x)  # (bs, input_dim, T, 1, 1)
        mu = self.model.stream_encode(x_in, first_chunk=first_chunk, scale=[0, 1])
        mu = self.postprocess(mu)  # (bs, T, z_dim)
        return mu

    @torch.no_grad()
    def stream_decode(self, mu, first_chunk=True):
        mu_in = self.preprocess(mu)  # (bs, z_dim, T, 1, 1)
        x_decoder = self.model.stream_decode(
            mu_in, first_chunk=first_chunk, scale=[0, 1]
        )
        x_out = self.postprocess(x_decoder)  # (bs, T, input_dim)
        x_out = x_out * self.std + self.mean
        return x_out

    def clear_cache(self):
        self.model.clear_cache()

    def generate(self, x):
        x = self._extract_inputs(x)
        features = x["feature"]
        feature_length = x["feature_length"]
        y_hat = self.decode(self.encode(features))

        y_hat_out = []

        for i in range(y_hat.shape[0]):
            # cut off the padding and align lengths
            valid_len = (
                feature_length[i] - 1
            ) // self.downsample_factor * self.downsample_factor + 1
            # Make sure both have the same length (take minimum)
            y_hat_out.append(y_hat[i, :valid_len])

        out = {}
        out["generated"] = y_hat_out
        return out
