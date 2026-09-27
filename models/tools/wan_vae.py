# This module uses modified code from Alibaba Wan Team
# Original source: https://github.com/Wan-Video/Wan2.2
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified to support 1d, 2d, 3d features with (B, C, T, 1, 1), (B, C, T, L, 1), (B, C, T, H, W) respectively.

import logging

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

CACHE_T = 2


class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolusion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)

        return super().forward(x)


class RMS_norm(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1)
        shape = (dim, *broadcastable_dims)

        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return F.normalize(x, dim=(1)) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):
    def forward(self, x):
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):
    def __init__(self, dim, mode, spatial_dim=2):
        assert mode in (
            "none",
            "upsample_temporal",
            "upsample_spatial",
            "upsample_temporal_spatial",
            "downsample_temporal",
            "downsample_spatial",
            "downsample_temporal_spatial",
        )
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == "upsample_temporal":
            self.resample = nn.Identity()
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "upsample_spatial" and spatial_dim == 2:
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
        elif mode == "upsample_spatial" and spatial_dim == 1:
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 1.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, (3, 1), padding=(1, 0)),
            )
        elif mode == "upsample_temporal_spatial" and spatial_dim == 2:
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "upsample_temporal_spatial" and spatial_dim == 1:
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 1.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, (3, 1), padding=(1, 0)),
            )
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample_temporal":
            self.resample = nn.Identity()
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
            )
        elif mode == "downsample_spatial" and spatial_dim == 2:
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2))
            )
        elif mode == "downsample_spatial" and spatial_dim == 1:
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 0, 0, 1)), nn.Conv2d(dim, dim, (3, 1), stride=(2, 1))
            )
        elif mode == "downsample_temporal_spatial" and spatial_dim == 2:
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2))
            )
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
            )
        elif mode == "downsample_temporal_spatial" and spatial_dim == 1:
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 0, 0, 1)), nn.Conv2d(dim, dim, (3, 1), stride=(2, 1))
            )
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
            )
        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == "upsample_temporal_spatial" or self.mode == "upsample_temporal":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if (
                        cache_x.shape[2] < 2
                        and feat_cache[idx] is not None
                        and feat_cache[idx] != "Rep"
                    ):
                        # cache last frame of last two chunk
                        cache_x = torch.cat(
                            [
                                feat_cache[idx][:, :, -1, :, :]
                                .unsqueeze(2)
                                .to(cache_x.device),
                                cache_x,
                            ],
                            dim=2,
                        )
                    if (
                        cache_x.shape[2] < 2
                        and feat_cache[idx] is not None
                        and feat_cache[idx] == "Rep"
                    ):
                        cache_x = torch.cat(
                            [torch.zeros_like(cache_x).to(cache_x.device), cache_x],
                            dim=2,
                        )
                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)

        if (
            self.mode == "downsample_temporal_spatial"
            or self.mode == "downsample_temporal"
        ):
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2)
                    )
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

    def init_weight(self, conv):
        conv_weight = conv.weight.detach().clone()
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        conv_weight.data[:, :, 1, 0, 0] = init_matrix  # * 0.5
        conv.weight = nn.Parameter(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        conv_weight = conv.weight.data.detach().clone()
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        conv_weight[: c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2 :, :, -1, 0, 0] = init_matrix
        conv.weight = nn.Parameter(conv_weight)
        nn.init.zeros_(conv.bias.data)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, spatial_dim=2, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.spatial_dim = spatial_dim

        if spatial_dim == 2:
            kernel_size = (3, 3, 3)
            padding = (1, 1, 1)
        elif spatial_dim == 1:
            kernel_size = (3, 3, 1)
            padding = (1, 1, 0)
        elif spatial_dim == 0:
            kernel_size = (3, 1, 1)
            padding = (1, 0, 0)
        else:
            kernel_size = (3, 3, 3)
            padding = (1, 1, 1)

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, kernel_size, padding=padding),
            RMS_norm(out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, kernel_size, padding=padding),
        )
        self.shortcut = (
            CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :]
                            .unsqueeze(2)
                            .to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = self.norm(x)
        x = rearrange(x, "b c t h w -> (b t) c h w")
        # compute query, key, value
        q, k, v = (
            self.to_qkv(x)
            .reshape(b * t, 1, c * 3, -1)
            .permute(0, 1, 3, 2)
            .contiguous()
            .chunk(3, dim=-1)
        )

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # apply attention
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
        )
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        # output
        x = self.proj(x)
        x = rearrange(x, "(b t) c h w-> b c t h w", t=t)
        return x + identity


def patchify(x, patch_size):
    if patch_size == 1:
        return x
    if x.dim() == 4:
        x = rearrange(x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size, r=patch_size)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b c f (h q) (w r) -> b (c r q) f h w",
            q=patch_size,
            r=patch_size,
        )
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")

    return x


def unpatchify(x, patch_size):
    if patch_size == 1:
        return x

    if x.dim() == 4:
        x = rearrange(x, "b (c r q) h w -> b c (h q) (w r)", q=patch_size, r=patch_size)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b (c r q) f h w -> b c f (h q) (w r)",
            q=patch_size,
            r=patch_size,
        )
    return x


class AvgDown3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        factor_t,
        factor_h=1,
        factor_w=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_h = factor_h
        self.factor_w = factor_w
        self.factor = self.factor_t * self.factor_h * self.factor_w

        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        pad = (0, 0, 0, 0, pad_t, 0)
        x = F.pad(x, pad)
        B, C, T, H, W = x.shape
        x = x.view(
            B,
            C,
            T // self.factor_t,
            self.factor_t,
            H // self.factor_h,
            self.factor_h,
            W // self.factor_w,
            self.factor_w,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(
            B,
            C * self.factor,
            T // self.factor_t,
            H // self.factor_h,
            W // self.factor_w,
        )
        x = x.view(
            B,
            self.out_channels,
            self.group_size,
            T // self.factor_t,
            H // self.factor_h,
            W // self.factor_w,
        )
        x = x.mean(dim=2)
        return x


class DupUp3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t,
        factor_h=1,
        factor_w=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.factor_t = factor_t
        self.factor_h = factor_h
        self.factor_w = factor_w
        self.factor = self.factor_t * self.factor_h * self.factor_w

        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x: torch.Tensor, first_chunk=False) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(
            x.size(0),
            self.out_channels,
            self.factor_t,
            self.factor_h,
            self.factor_w,
            x.size(2),
            x.size(3),
            x.size(4),
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(
            x.size(0),
            self.out_channels,
            x.size(2) * self.factor_t,
            x.size(4) * self.factor_h,
            x.size(6) * self.factor_w,
        )
        if first_chunk:
            x = x[:, :, self.factor_t - 1 :, :, :]
        return x


class Down_ResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dropout,
        mult,
        temperal_downsample=False,
        spatial_downsample=False,
        spatial_dim=2,
    ):
        super().__init__()

        # Determine spatial factors based on spatial_downsample
        down_flag = temperal_downsample or spatial_downsample
        factor_h, factor_w = 1, 1
        if spatial_downsample:
            if spatial_dim == 2:
                factor_h, factor_w = 2, 2
            elif spatial_dim == 1:
                factor_h, factor_w = 2, 1

        # Shortcut path with downsample
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_h=factor_h,
            factor_w=factor_w,
        )

        # Main path with residual blocks and downsample
        downsamples = []
        for _ in range(mult):
            downsamples.append(ResidualBlock(in_dim, out_dim, spatial_dim, dropout))
            in_dim = out_dim

        # Add the final downsample block
        if down_flag:
            if temperal_downsample and spatial_downsample and spatial_dim > 0:
                mode = "downsample_temporal_spatial"
            elif temperal_downsample:
                mode = "downsample_temporal"
            elif spatial_downsample and spatial_dim > 0:
                mode = "downsample_spatial"
            else:
                mode = "none"
            downsamples.append(Resample(out_dim, mode=mode, spatial_dim=spatial_dim))

        self.downsamples = nn.Sequential(*downsamples)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        x_copy = x.clone()
        for module in self.downsamples:
            x = module(x, feat_cache, feat_idx)

        return x + self.avg_shortcut(x_copy)


class Up_ResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dropout,
        mult,
        temperal_upsample=False,
        spatial_upsample=False,
        spatial_dim=2,
    ):
        super().__init__()

        # Determine spatial factors based on spatial_upsample
        up_flag = temperal_upsample or spatial_upsample
        factor_h, factor_w = 1, 1
        if spatial_upsample:
            if spatial_dim == 2:
                factor_h, factor_w = 2, 2
            elif spatial_dim == 1:
                factor_h, factor_w = 2, 1

        # Shortcut path with upsample
        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_h=factor_h,
                factor_w=factor_w,
            )
        else:
            self.avg_shortcut = None

        # Main path with residual blocks and upsample
        upsamples = []
        for _ in range(mult):
            upsamples.append(ResidualBlock(in_dim, out_dim, spatial_dim, dropout))
            in_dim = out_dim

        # Add the final upsample block
        if up_flag:
            if temperal_upsample and spatial_upsample and spatial_dim > 0:
                mode = "upsample_temporal_spatial"
            elif temperal_upsample:
                mode = "upsample_temporal"
            elif spatial_upsample and spatial_dim > 0:
                mode = "upsample_spatial"
            else:
                mode = "none"
            upsamples.append(Resample(out_dim, mode=mode, spatial_dim=spatial_dim))

        self.upsamples = nn.Sequential(*upsamples)

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        x_main = x.clone()
        for module in self.upsamples:
            x_main = module(x_main, feat_cache, feat_idx)
        if self.avg_shortcut is not None:
            x_shortcut = self.avg_shortcut(x, first_chunk)
            return x_main + x_shortcut
        else:
            return x_main


class Encoder3d(nn.Module):
    def __init__(
        self,
        input_dim=12,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        spatial_downsample=[True, True, True],
        spatial_dim=2,
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        if spatial_dim == 2:
            kernel_size = (3, 3, 3)
            padding = (1, 1, 1)
        elif spatial_dim == 1:
            kernel_size = (3, 3, 1)
            padding = (1, 1, 0)
        elif spatial_dim == 0:
            kernel_size = (3, 1, 1)
            padding = (1, 0, 0)
        else:
            kernel_size = (3, 3, 3)
            padding = (1, 1, 1)

        # init block
        self.conv1 = CausalConv3d(input_dim, dims[0], kernel_size, padding=padding)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_down_flag = (
                temperal_downsample[i] if i < len(temperal_downsample) else False
            )
            spatial_down_flag = (
                spatial_downsample[i] if i < len(spatial_downsample) else False
            )
            downsamples.append(
                Down_ResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks,
                    temperal_downsample=t_down_flag,
                    spatial_downsample=spatial_down_flag,
                    spatial_dim=spatial_dim,
                )
            )
            scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        if spatial_dim > 0:
            self.middle = nn.Sequential(
                ResidualBlock(
                    out_dim, out_dim, spatial_dim=spatial_dim, dropout=dropout
                ),
                AttentionBlock(out_dim),
                ResidualBlock(
                    out_dim, out_dim, spatial_dim=spatial_dim, dropout=dropout
                ),
            )
        else:
            self.middle = nn.Sequential(
                ResidualBlock(
                    out_dim, out_dim, spatial_dim=spatial_dim, dropout=dropout
                ),
                RMS_norm(out_dim),
                CausalConv3d(out_dim, out_dim, 1),
                ResidualBlock(
                    out_dim, out_dim, spatial_dim=spatial_dim, dropout=dropout
                ),
            )

        # # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim),
            nn.SiLU(),
            CausalConv3d(out_dim, z_dim, kernel_size, padding=padding),
        )

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :]
                            .unsqueeze(2)
                            .to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)

        return x


class Decoder3d(nn.Module):
    def __init__(
        self,
        output_dim=12,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
        spatial_upsample=[True, True, True],
        spatial_dim=2,
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample
        self.spatial_upsample = spatial_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)
        if spatial_dim == 2:
            kernel_size = (3, 3, 3)
            padding = (1, 1, 1)
        elif spatial_dim == 1:
            kernel_size = (3, 3, 1)
            padding = (1, 1, 0)
        elif spatial_dim == 0:
            kernel_size = (3, 1, 1)
            padding = (1, 0, 0)
        else:
            kernel_size = (3, 3, 3)
            padding = (1, 1, 1)
        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], kernel_size, padding=padding)

        # middle blocks
        if spatial_dim > 0:
            self.middle = nn.Sequential(
                ResidualBlock(
                    dims[0], dims[0], spatial_dim=spatial_dim, dropout=dropout
                ),
                AttentionBlock(dims[0]),
                ResidualBlock(
                    dims[0], dims[0], spatial_dim=spatial_dim, dropout=dropout
                ),
            )
        else:
            self.middle = nn.Sequential(
                ResidualBlock(
                    dims[0], dims[0], spatial_dim=spatial_dim, dropout=dropout
                ),
                RMS_norm(dims[0]),
                CausalConv3d(dims[0], dims[0], 1),
                ResidualBlock(
                    dims[0], dims[0], spatial_dim=spatial_dim, dropout=dropout
                ),
            )

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_up_flag = temperal_upsample[i] if i < len(temperal_upsample) else False
            spatial_up_flag = (
                spatial_upsample[i] if i < len(spatial_upsample) else False
            )
            upsamples.append(
                Up_ResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks + 1,
                    temperal_upsample=t_up_flag,
                    spatial_upsample=spatial_up_flag,
                    spatial_dim=spatial_dim,
                )
            )
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim),
            nn.SiLU(),
            CausalConv3d(out_dim, output_dim, kernel_size, padding=padding),
        )

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx, first_chunk)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :]
                            .unsqueeze(2)
                            .to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class WanVAE_(nn.Module):
    def __init__(
        self,
        input_dim=12,
        dim=160,
        dec_dim=256,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        spatial_downsample=[True, True, True],
        spatial_dim=2,
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.spatial_downsample = spatial_downsample
        self.temperal_upsample = temperal_downsample[::-1]
        self.spatial_upsample = spatial_downsample[::-1]

        # modules
        self.encoder = Encoder3d(
            input_dim,
            dim,
            z_dim * 2,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_downsample,
            self.spatial_downsample,
            spatial_dim,
            dropout,
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(
            input_dim,
            dec_dim,
            z_dim,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_upsample,
            self.spatial_upsample,
            spatial_dim,
            dropout,
        )

        # Conv counts are architecture-static, so compute once.
        self._enc_conv_num = count_conv3d(self.encoder)
        self._conv_num = count_conv3d(self.decoder)
        self._enc_feat_map = [None] * self._enc_conv_num
        self._feat_map = [None] * self._conv_num

    def forward(self, x, scale=[0, 1]):
        mu = self.encode(x, scale)
        x_recon = self.decode(mu, scale)
        return x_recon, mu

    def encode(self, x, scale, patch_size=1, return_dist=False):
        self.clear_cache()
        x = patchify(x, patch_size=patch_size)
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
                out = torch.cat([out, out_], 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1
            )
        else:
            mu = (mu - scale[0]) * scale[1]
        self.clear_cache()

        if return_dist:
            return mu, log_var
        else:
            return mu

    def decode(self, z, scale, patch_size=1):
        self.clear_cache()
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1
            )
        else:
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i : i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                    first_chunk=True,
                )
            else:
                out_ = self.decoder(
                    x[:, :, i : i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                )
                out = torch.cat([out, out_], 2)
        out = unpatchify(out, patch_size=patch_size)
        self.clear_cache()
        return out

    @torch.no_grad()
    def stream_encode(self, x, first_chunk, scale, patch_size=1, return_dist=False):
        if first_chunk:
            self.clear_enc_cache()
        x = patchify(x, patch_size=patch_size)
        t = x.shape[2]
        if first_chunk:
            iter_ = 1 + (t - 1) // 4
        else:
            iter_ = t // 4
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                if first_chunk:
                    out = self.encoder(
                        x[:, :, :1, :, :],
                        feat_cache=self._enc_feat_map,
                        feat_idx=self._enc_conv_idx,
                    )
                else:
                    out = self.encoder(
                        x[:, :, :4, :, :],
                        feat_cache=self._enc_feat_map,
                        feat_idx=self._enc_conv_idx,
                    )
            else:
                if first_chunk:
                    out_ = self.encoder(
                        x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
                        feat_cache=self._enc_feat_map,
                        feat_idx=self._enc_conv_idx,
                    )
                else:
                    out_ = self.encoder(
                        x[:, :, 4 * i : 4 * (i + 1), :, :],
                        feat_cache=self._enc_feat_map,
                        feat_idx=self._enc_conv_idx,
                    )
                out = torch.cat([out, out_], 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1
            )
        else:
            mu = (mu - scale[0]) * scale[1]

        if return_dist:
            return mu, log_var
        else:
            return mu

    @torch.no_grad()
    def stream_decode(self, z, first_chunk, scale, patch_size=1):
        if first_chunk:
            self.clear_dec_cache()
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i : i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                    first_chunk=first_chunk,
                )
            else:
                out_ = self.decoder(
                    x[:, :, i : i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                )
                out = torch.cat([out, out_], 2)
        out = unpatchify(out, patch_size=patch_size)
        return out

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, features, deterministic=False):
        mu, log_var = self.encode(features, return_dist=True)
        if deterministic:
            return mu
        else:
            return self.reparameterize(mu, log_var)

    def clear_enc_cache(self):
        self._enc_feat_map = [None] * self._enc_conv_num

    def clear_dec_cache(self):
        self._feat_map = [None] * self._conv_num

    def clear_cache(self):
        self.clear_enc_cache()
        self.clear_dec_cache()
