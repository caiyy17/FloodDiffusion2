"""WAN VAE with an explicit per-latent root-motion condition.

The encoder consumes the complete motion feature and produces the configured
``z_dim`` stochastic latent.  The decoder receives four frames of root motion
followed by that latent and reconstructs the complete feature.  Root motion is
only the representation's XZ difference plus angular difference; pelvis/root
pose channels are deliberately excluded.  WAN's temporal layout is causal and
asymmetric:

* latent 0 decodes frame 0;
* latent j >= 1 decodes frames ``1 + 4*(j-1) : 1 + 4*j``.

Consequently the first root condition repeats frame 0 four times.  This is
also what makes full and streaming decode use exactly the same alignment.
"""

import numpy as np
import torch

from visualization import registry as representation_registry

from .tools.wan_vae import (
    CausalConv3d,
    Decoder3d,
    WanVAE_,
    count_conv3d,
    unpatchify,
)
from .vae_wan import VAEWanModel
from .quadratic_loss import load_quadratic_loss_matrix, quadratic_error_sum


# Keep in sync with models.diffusion_forcing_position_wan.
STD_NORMALIZATION_FLOOR = 1e-3
WAN_ROOT_GROUP_SIZE = 4


# This is a position-model contract, intentionally kept out of the shared
# visualization registry: all four representations store the condition as a
# contiguous prefix, but MotionStreamer uses a 6D angular difference.
POSITION_ROOT_DIMS = {
    "humanml3d263": 3,       # angular diff 1 + XZ diff 2
    "mei138": 3,             # angular diff 1 + XZ diff 2
    "somarelative271": 3,    # angular diff 1 + XZ diff 2
    "motionstreamer272": 8,  # XZ diff 2 + angular diff 6D
}


def resolve_position_representation(
    representation=None,
    *,
    feature_dim=None,
    root_dim=None,
):
    """Resolve and validate the root-prefix contract for a representation.

    ``representation`` names the position root contract.  A custom feature
    width without a representation remains usable when ``root_dim`` is given
    explicitly (and retains the 3D default otherwise).
    """
    name = None
    if representation is not None:
        name = representation_registry.canonical(representation)

    expected_root_dim = None
    if name is not None:
        if name not in POSITION_ROOT_DIMS:
            raise ValueError(
                f"representation {name!r} has no position root contract; "
                f"supported: {sorted(POSITION_ROOT_DIMS)}"
            )
        expected_feature_dim = representation_registry.dim(name)
        if feature_dim is not None and int(feature_dim) != expected_feature_dim:
            raise ValueError(
                f"representation {name!r} requires feature_dim="
                f"{expected_feature_dim}, got {int(feature_dim)}"
            )
        expected_root_dim = POSITION_ROOT_DIMS[name]

    if root_dim is None:
        resolved_root_dim = expected_root_dim if expected_root_dim is not None else 3
    else:
        resolved_root_dim = int(root_dim)
        if expected_root_dim is not None and resolved_root_dim != expected_root_dim:
            raise ValueError(
                f"representation {name!r} requires root_dim={expected_root_dim}, "
                f"got {resolved_root_dim}"
            )
    if resolved_root_dim <= 0:
        raise ValueError(f"root_dim must be positive, got {resolved_root_dim}")
    if feature_dim is not None and resolved_root_dim >= int(feature_dim):
        raise ValueError(
            f"root_dim must be smaller than feature_dim={int(feature_dim)}, "
            f"got {resolved_root_dim}"
        )
    return name, resolved_root_dim


def guard_std(std, floor=STD_NORMALIZATION_FLOOR):
    """Replace near-zero standard deviations with 1 (i.e. do not divide)."""
    std = torch.as_tensor(std).float()
    return torch.where(std < floor, torch.ones_like(std), std)


def _as_root_frames(position, root_dim=3):
    """Return ``position`` as ``(B, T, root_dim)`` raw root values."""
    if not torch.is_tensor(position):
        position = torch.as_tensor(position)
    if position.ndim == 2:
        position = position.unsqueeze(0)
    if position.ndim != 3:
        raise ValueError(
            f"position must have shape (B,T,C) or (T,C), got {tuple(position.shape)}"
        )
    if position.size(-1) < root_dim:
        raise ValueError(
            f"position last dimension must be at least {root_dim}, "
            f"got {position.size(-1)}"
        )
    return position[..., :root_dim]


def pack_root_condition(
    position,
    latent_length=None,
    *,
    root_dim=3,
    group_size=4,
    first_chunk=True,
):
    """Pack raw per-frame root values into one condition per latent.

    Args:
        position: ``(B,T,>=root_dim)`` or ``(T,>=root_dim)``.  Passing an
            already-packed ``(..., root_dim * group_size)`` tensor is supported
            when its time dimension equals ``latent_length``.
        latent_length: requested number of latent conditions.  Missing tail
            frames are replicate-padded; extra frames are ignored.
        first_chunk: WAN's first decode chunk emits one frame for its first
            latent.  Later streaming chunks emit four frames per latent.
    """
    if not torch.is_tensor(position):
        position = torch.as_tensor(position)
    if position.ndim == 2:
        position = position.unsqueeze(0)

    packed_dim = root_dim * group_size
    if position.ndim == 3 and position.size(-1) == packed_dim:
        if latent_length is not None and position.size(1) != int(latent_length):
            raise ValueError(
                f"packed position has {position.size(1)} tokens, expected "
                f"{int(latent_length)}"
            )
        return position

    roots = _as_root_frames(position, root_dim=root_dim)
    batch_size, frame_count, _ = roots.shape
    if frame_count == 0:
        if latent_length not in (None, 0):
            raise ValueError("cannot build non-empty root condition from zero frames")
        return roots.new_empty(batch_size, 0, packed_dim)

    if latent_length is None:
        if first_chunk:
            latent_length = 1 + max(frame_count - 1, 0) // group_size
        else:
            latent_length = frame_count // group_size
    latent_length = int(latent_length)
    if latent_length < 0:
        raise ValueError(f"latent_length must be non-negative, got {latent_length}")
    if latent_length == 0:
        return roots.new_empty(batch_size, 0, packed_dim)

    token_ids = torch.arange(latent_length, device=roots.device)[:, None]
    offsets = torch.arange(group_size, device=roots.device)[None, :]
    if first_chunk:
        # token 0 is [r0,r0,r0,r0]; token j>=1 is the next causal group.
        indices = 1 + (token_ids - 1) * group_size + offsets
        indices[0] = 0
    else:
        indices = token_ids * group_size + offsets
    indices = indices.clamp(min=0, max=frame_count - 1)
    packed = roots[:, indices.reshape(-1), :]
    return packed.reshape(batch_size, latent_length, packed_dim)


def unpack_root_condition(
    packed,
    *,
    root_dim=3,
    group_size=4,
    first_chunk=True,
    output_length=None,
):
    """Inverse the WAN-aligned packing for decoder output clamping."""
    if not torch.is_tensor(packed):
        packed = torch.as_tensor(packed)
    if packed.ndim == 2:
        packed = packed.unsqueeze(0)
    expected = root_dim * group_size
    if packed.ndim != 3 or packed.size(-1) != expected:
        raise ValueError(
            f"packed root must have shape (B,L,{expected}), got {tuple(packed.shape)}"
        )
    batch_size, latent_length, _ = packed.shape
    groups = packed.reshape(batch_size, latent_length, group_size, root_dim)
    if latent_length == 0:
        roots = packed.new_empty(batch_size, 0, root_dim)
    elif first_chunk:
        roots = torch.cat(
            [groups[:, 0, :1, :], groups[:, 1:, :, :].flatten(1, 2)],
            dim=1,
        )
    else:
        roots = groups.flatten(1, 2)
    if output_length is not None:
        roots = roots[:, : int(output_length)]
    return roots


class WanVAEPosition_(WanVAE_):
    """WanVAE_ with a root-conditioned decoder and unchanged encoder latent."""

    def __init__(self, *args, position_dim=12, **kwargs):
        super().__init__(*args, **kwargs)
        self.position_dim = int(position_dim)
        self.decoder_input_dim = self.position_dim + self.z_dim

        # Only the decoder entry changes.  Encoder, mu/logvar, and KL latent
        # dimensionality stay exactly at z_dim.
        self.conv2 = CausalConv3d(
            self.decoder_input_dim, self.decoder_input_dim, 1
        )
        self.decoder = Decoder3d(
            kwargs.get("input_dim", 12),
            kwargs.get("dec_dim", 256),
            self.decoder_input_dim,
            kwargs.get("dim_mult", [1, 2, 4, 4]),
            kwargs.get("num_res_blocks", 2),
            kwargs.get("attn_scales", []),
            self.temperal_upsample,
            self.spatial_upsample,
            kwargs.get("spatial_dim", 2),
            kwargs.get("dropout", 0.0),
        )
        self._conv_num = count_conv3d(self.decoder)
        self._feat_map = [None] * self._conv_num

    def _validate_decoder_input(self, z):
        if z.size(1) != self.decoder_input_dim:
            raise ValueError(
                f"position decoder expects {self.decoder_input_dim} channels "
                f"({self.position_dim} condition + {self.z_dim} latent), "
                f"got {z.size(1)}"
            )

    def decode(self, z, scale=[0, 1], patch_size=1):
        self.clear_cache()
        self._validate_decoder_input(z)
        if isinstance(scale[0], torch.Tensor):
            if scale[0].numel() != self.decoder_input_dim:
                raise ValueError(
                    "tensor decoder scale must cover all "
                    f"{self.decoder_input_dim} channels"
                )
            z = z / scale[1].view(1, -1, 1, 1, 1) + scale[0].view(
                1, -1, 1, 1, 1
            )
        else:
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i : i + 1],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                    first_chunk=True,
                )
            else:
                out_i = self.decoder(
                    x[:, :, i : i + 1],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                )
                out = torch.cat([out, out_i], dim=2)
        out = unpatchify(out, patch_size=patch_size)
        self.clear_cache()
        return out

    @torch.no_grad()
    def stream_decode(self, z, first_chunk, scale=[0, 1], patch_size=1):
        if first_chunk:
            self.clear_dec_cache()
        self._validate_decoder_input(z)
        if isinstance(scale[0], torch.Tensor):
            if scale[0].numel() != self.decoder_input_dim:
                raise ValueError(
                    "tensor decoder scale must cover all "
                    f"{self.decoder_input_dim} channels"
                )
            z = z / scale[1].view(1, -1, 1, 1, 1) + scale[0].view(
                1, -1, 1, 1, 1
            )
        else:
            z = z / scale[1] + scale[0]
        x = self.conv2(z)
        for i in range(z.shape[2]):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i : i + 1],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                    first_chunk=first_chunk,
                )
            else:
                out_i = self.decoder(
                    x[:, :, i : i + 1],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                )
                out = torch.cat([out, out_i], dim=2)
        return unpatchify(out, patch_size=patch_size)


class VAEWanPositionModel(VAEWanModel):
    """Full-feature WAN VAE whose decoder is conditioned on root motion."""

    requires_position_condition = True

    def __init__(
        self,
        *args,
        representation=None,
        root_dim=None,
        root_group_size=WAN_ROOT_GROUP_SIZE,
        **kwargs,
    ):
        loss_W = kwargs.pop("loss_W", None)
        loss_w_coefficient = kwargs.pop("loss_w_coefficient", 1.0)
        # The base VAE sees the complete 138-D feature, whereas the position
        # objective scores only the non-root state.  Load the 135-D matrix
        # after resolving root_dim below.
        super().__init__(*args, loss_W=None, **kwargs)
        self.representation, self.root_dim = resolve_position_representation(
            representation,
            feature_dim=self.input_dim,
            root_dim=root_dim,
        )
        self.root_group_size = int(root_group_size)
        self.loss_matrix = load_quadratic_loss_matrix(
            loss_W, loss_w_coefficient, self.input_dim - self.root_dim
        )
        if self.root_group_size != WAN_ROOT_GROUP_SIZE:
            raise ValueError(
                "the WAN position ABI groups exactly "
                f"{WAN_ROOT_GROUP_SIZE} root frames per latent, got "
                f"root_group_size={self.root_group_size}"
            )
        if self.root_group_size != self.downsample_factor:
            raise ValueError(
                "root_group_size must equal the WAN temporal downsample factor "
                f"({self.downsample_factor}), got {self.root_group_size}"
            )
        self.position_dim = self.root_dim * self.root_group_size
        self.conditioned_latent_dim = self.position_dim + self.z_dim
        self.model = WanVAEPosition_(
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
            position_dim=self.position_dim,
        )

    @property
    def root_mean(self):
        return self.mean[: self.root_dim]

    @property
    def root_std(self):
        return self.std[: self.root_dim]

    def _normalize_packed_root(self, packed):
        mean = self.root_mean.repeat(self.root_group_size)
        # Same rule as the diffusion models: a channel that is constant in the
        # ground truth has std ~0, and dividing by it would amplify float noise
        # instead of normalizing.
        std = guard_std(self.root_std).repeat(self.root_group_size)
        return (packed - mean) / std

    def _split_decoder_inputs(self, latent, position, *, first_chunk):
        if latent.ndim != 3:
            raise ValueError(
                f"latent must have shape (B,L,C), got {tuple(latent.shape)}"
            )
        if latent.size(-1) == self.conditioned_latent_dim:
            if position is not None:
                raise ValueError(
                    "position must be omitted when latent already contains "
                    f"{self.conditioned_latent_dim}D "
                    f"[root{self.position_dim}, latent{self.z_dim}]"
                )
            packed_raw = latent[..., : self.position_dim]
            latent = latent[..., self.position_dim :]
        elif latent.size(-1) == self.z_dim:
            if position is None:
                raise ValueError(
                    "position/root input is required when decoding a "
                    f"{self.z_dim}D latent"
                )
            packed_raw = pack_root_condition(
                position,
                latent_length=latent.size(1),
                root_dim=self.root_dim,
                group_size=self.root_group_size,
                first_chunk=first_chunk,
            ).to(device=latent.device, dtype=latent.dtype)
        else:
            raise ValueError(
                f"latent last dimension must be {self.z_dim} or "
                f"{self.conditioned_latent_dim}, got {latent.size(-1)}"
            )
        packed_norm = self._normalize_packed_root(packed_raw)
        decoder_input = torch.cat([packed_norm, latent], dim=-1)
        return decoder_input, packed_raw

    def forward(self, x):
        x = self._extract_inputs(x)
        raw_features = x["feature"]
        feature_length = x["feature_length"]
        features = (raw_features - self.mean) / self.std
        batch_size, seq_len = features.shape[:2]
        mask = torch.arange(seq_len, device=features.device)[None, :] < (
            feature_length[:, None]
        )

        x_in = self.preprocess(features)
        mu, log_var = self.model.encode(x_in, scale=[0, 1], return_dist=True)
        z = self.model.reparameterize(mu, log_var)
        latent_length = z.size(2)
        packed_raw = pack_root_condition(
            raw_features[..., : self.root_dim],
            latent_length=latent_length,
            root_dim=self.root_dim,
            group_size=self.root_group_size,
            first_chunk=True,
        )
        packed_norm = self._normalize_packed_root(packed_raw)
        packed_norm = self.preprocess(packed_norm)
        decoder_input = torch.cat([packed_norm, z], dim=1)
        x_decoder = self.model.decode(decoder_input, scale=[0, 1])
        x_out = self.postprocess(x_decoder)

        if x_out.size(1) != features.size(1):
            min_len = min(x_out.size(1), features.size(1))
            x_out = x_out[:, :min_len]
            features = features[:, :min_len]
            mask = mask[:, :min_len]

        mask_expanded = mask
        for _ in range(features.ndim - 2):
            mask_expanded = mask_expanded.unsqueeze(-1)
        spatial_numel = int(np.prod(features.shape[2:-1])) if features.ndim > 3 else 1
        loss_per_element = self.RECONS_LOSS(x_out, features)
        root_weights = self.recons_weights[: self.root_dim]
        root_loss = (
            loss_per_element[..., : self.root_dim]
            * mask_expanded
            * root_weights
        ).sum() / (mask_expanded.sum() * spatial_numel * root_weights.sum())
        if self.loss_matrix is None:
            # Preserve the original full-feature Smooth L1 objective.
            denom = mask_expanded.sum() * spatial_numel * self.recons_weights.sum()
            loss_recons = (
                loss_per_element * mask_expanded * self.recons_weights
            ).sum() / denom
        else:
            pose_error = (
                x_out[..., self.root_dim :] - features[..., self.root_dim :]
            ) * mask_expanded
            pose_dim = self.input_dim - self.root_dim
            loss_recons = quadratic_error_sum(
                pose_error.movedim(-1, 0), self.loss_matrix
            ) / (mask.sum() * spatial_numel * pose_dim)

        latent_mask = torch.arange(mu.size(2), device=features.device)[None, :] < (
            (feature_length + self.downsample_factor - 1)
            // self.downsample_factor
        )[:, None]
        kl_per_element = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
        kl_mask = latent_mask[:, None, :, None, None]
        latent_elements = mu.size(1) * mu.size(3) * mu.size(4)
        kl_loss = (kl_per_element * kl_mask).sum() / (
            latent_mask.sum() * latent_elements
        )
        total_loss = self.LAMBDA_FEATURE * loss_recons + self.LAMBDA_KL * kl_loss
        return {
            "total": total_loss,
            "recons": loss_recons,
            "root_recons": root_loss,
            "kl": kl_loss,
        }

    def encode_latent(self, x):
        """Encode the complete motion feature to the configured latent."""
        return super().encode(x)

    def encode(self, x):
        """Alias kept explicit: return only the configured latent channels."""
        return self.encode_latent(x)

    def encode_with_position(self, x):
        """Return ``[packed raw root motion, latent]`` tokens."""
        latent = self.encode_latent(x)
        packed = pack_root_condition(
            x[..., : self.root_dim],
            latent_length=latent.size(1),
            root_dim=self.root_dim,
            group_size=self.root_group_size,
            first_chunk=True,
        ).to(device=latent.device, dtype=latent.dtype)
        return torch.cat([packed, latent], dim=-1)

    def decode(self, latent, position=None, clamp_position=True):
        decoder_input, packed_raw = self._split_decoder_inputs(
            latent, position, first_chunk=True
        )
        decoder_in = self.preprocess(decoder_input)
        x_decoder = self.model.decode(decoder_in, scale=[0, 1])
        x_out = self.postprocess(x_decoder) * self.std + self.mean
        if clamp_position:
            roots = unpack_root_condition(
                packed_raw,
                root_dim=self.root_dim,
                group_size=self.root_group_size,
                first_chunk=True,
                output_length=x_out.size(1),
            )
            x_out = x_out.clone()
            x_out[..., : self.root_dim] = roots.to(x_out)
        return x_out

    @torch.no_grad()
    def stream_encode(self, x, first_chunk=True):
        return super().stream_encode(x, first_chunk=first_chunk)

    @torch.no_grad()
    def stream_encode_with_position(self, x, first_chunk=True):
        latent = self.stream_encode(x, first_chunk=first_chunk)
        packed = pack_root_condition(
            x[..., : self.root_dim],
            latent_length=latent.size(1),
            root_dim=self.root_dim,
            group_size=self.root_group_size,
            first_chunk=first_chunk,
        ).to(device=latent.device, dtype=latent.dtype)
        return torch.cat([packed, latent], dim=-1)

    @torch.no_grad()
    def stream_decode(
        self, latent, position=None, first_chunk=True, clamp_position=True
    ):
        decoder_input, packed_raw = self._split_decoder_inputs(
            latent, position, first_chunk=first_chunk
        )
        decoder_in = self.preprocess(decoder_input)
        x_decoder = self.model.stream_decode(
            decoder_in, first_chunk=first_chunk, scale=[0, 1]
        )
        x_out = self.postprocess(x_decoder) * self.std + self.mean
        if clamp_position:
            roots = unpack_root_condition(
                packed_raw,
                root_dim=self.root_dim,
                group_size=self.root_group_size,
                first_chunk=first_chunk,
                output_length=x_out.size(1),
            )
            x_out = x_out.clone()
            x_out[..., : self.root_dim] = roots.to(x_out)
        return x_out

    def generate(self, x):
        inputs = self._extract_inputs(x)
        features = inputs["feature"]
        feature_length = inputs["feature_length"]
        latent = self.encode_latent(features)
        y_hat = self.decode(latent, features[..., : self.root_dim])
        generated = []
        for i in range(y_hat.size(0)):
            valid_len = (
                (feature_length[i] - 1) // self.downsample_factor
                * self.downsample_factor
                + 1
            )
            generated.append(y_hat[i, :valid_len])
        return {"generated": generated}


# Short alias matching the naming style used by some local configs.
VAEWanPosition = VAEWanPositionModel
