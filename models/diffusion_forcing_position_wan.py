"""Position-conditioned diffusion-forcing models with an isolated engine.

The original models.diffusion_forcing_wan module is intentionally not modified.
This module carries its own position-capable engine so direct per-frame root
motion and WAN-packed root conditioning can share identical offline/stream
logic without changing the base DiffForcingWanModel behavior.  Root motion means
only XZ difference plus angular difference; pelvis/root pose remains in the
diffused pose state.
"""

import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .tools.t5 import T5EncoderModel
from .tools.wan_model import WanModel
from .normalization import load_statistics
from .quadratic_loss import load_quadratic_loss_matrix, quadratic_error_sum

EPSILON = 0.05
# Most-negative rope frame index supported by rope_params_with_negative(1024, ...).
# Clamping at this value keeps trim_stream's shifted ids inside the cached range.
ROPE_MIN_INDEX = -1024


class TriangularTimeScheduler:
    def __init__(self, config):
        self.steps = config["steps"]
        self.chunk_size = config["chunk_size"]
        self.random_epsilon = config.get("random_epsilon", 0.00)  # schedule jittering
        self.noise_type = config.get("noise_type", "linear")
        self.sigma_type = config.get("sigma_type", "zero")  # "zero", "memoryless"

        if self.noise_type == "exponential" or self.noise_type == "exponential_rev":
            self.exp_max = config.get("exp_max", 5.0)
        elif self.noise_type == "diffusion":
            self.T = config.get("T", 1000)
            self.beta_start = config.get("beta_start", 0.0001)
            self.beta_end = config.get("beta_end", 0.02)

        if self.sigma_type == "memoryless":
            self.sigma_scale = config.get("sigma_scale", 1.0)
        self.content_len = config.get("content_len", None)
        # True (default): t_max covers the tail ramp-down in place (pairs
        # with extra_len=0). False: original L/c; cover the tail with
        # extra_len padding (typically chunk_size-1) at generate time.
        self.t_max_tail = bool(config.get("t_max_tail", True))
        # For simplicity we require steps to be divisible by chunk_size, so that time windows align well.

    def _t_max_exact(self, valid_len):
        """Exact rational t upper bound (single source of truth).
        t_max_tail: 1 + (L-1)/c so the last frame's band is covered in
        place; otherwise the original L/c (tail covered via extra_len)."""
        if self.t_max_tail:
            return Fraction(valid_len + self.chunk_size - 1, self.chunk_size)
        return Fraction(valid_len, self.chunk_size)

    def get_total_steps(self, seq_len):
        # Exact arithmetic: deriving via float double-rounds and can lose the
        # final step (e.g. steps=30, c=30, L=94).
        total = self.steps * self._t_max_exact(seq_len)
        assert total.denominator == 1, (
            f"inference grid does not close: steps={self.steps} * t_max"
            f"({seq_len}+{self.chunk_size}-1)/{self.chunk_size} is not an "
            f"integer; use steps divisible by chunk_size"
        )
        return int(total)

    def get_t_max(self, valid_len):
        """Float view of _t_max_exact (for continuous t sampling)."""
        return float(self._t_max_exact(valid_len))

    def get_time_steps(self, device, valid_len, current_step):
        """current_step (int) → one t scalar shared by the whole batch,
        returned as a per-sample List. Inference-only (generate / stream);
        training's stratified sampling bypasses this and calls the
        lower-level helpers directly."""
        t = current_step * (1 / self.steps)
        return [torch.tensor(t, device=device) for _ in range(len(valid_len))]

    def get_time_schedules(self, device, valid_len, time_steps, training=False):
        time_schedules = []
        time_schedules_derivative = []
        for i in range(len(valid_len)):
            t = time_steps[i].item()
            current_time_schedules = torch.clamp(
                -torch.arange(valid_len[i], device=device) / self.chunk_size + t,
                min=0.0,
                max=1.0,
            )
            current_time_schedules_derivative = torch.ones_like(
                current_time_schedules
            ) * (1 / self.steps)
            if training:
                current_time_schedules = torch.clamp(
                    current_time_schedules
                    + torch.randn_like(current_time_schedules) * self.random_epsilon,
                    min=0.0,
                    max=1.0,
                )
            time_schedules.append(current_time_schedules)
            time_schedules_derivative.append(current_time_schedules_derivative)
        return time_schedules, time_schedules_derivative

    def get_windows(self, valid_len, time_steps, training=False):
        # Grid t (inference) lands exactly on band boundaries; +eps makes the
        # half-open [0, 1) level convention robust to float rounding there.
        # Continuous t (training) never hits boundaries; eps would only shift
        # the band by eps/chunk_size, so it is skipped.
        eps = 0.0 if training else 0.5 * (1 / (self.steps * self.chunk_size))
        input_start, input_end, output_start, output_end = [], [], [], []
        for i in range(len(time_steps)):
            t = time_steps[i].item()
            start_index = max(
                0,
                math.floor((t - 1) * self.chunk_size + eps) + 1,
            )
            end_index = min(
                valid_len[i],
                math.floor(t * self.chunk_size + eps) + 1,
            )
            # An exact boundary hit at the sequence end gives start == end;
            # the last frame is still in the band, so fall back to [end-1, end).
            start_index = min(start_index, max(end_index - 1, 0))

            if self.content_len is not None:
                input_start.append(max(0, end_index - self.content_len))
            else:
                input_start.append(0)
            input_end.append(end_index)
            output_start.append(start_index)
            output_end.append(end_index)
        return input_start, input_end, output_start, output_end

    def get_noise_levels(self, device, valid_len, time_schedules):
        alpha = []
        dalpha = []
        dlog_alpha = []
        beta = []
        dbeta = []
        dlog_beta = []
        sigma = []
        for i in range(len(valid_len)):
            t = time_schedules[i]
            if self.noise_type == "linear":
                alpha_i = t
                dalpha_i = torch.ones_like(alpha_i)
                dlog_alpha_i = dalpha_i / torch.clamp(alpha_i, min=EPSILON)
                beta_i = 1 - t
                dbeta_i = -torch.ones_like(beta_i)
                dlog_beta_i = dbeta_i / torch.clamp(beta_i, min=EPSILON)
            elif self.noise_type == "exponential":
                # "eps" prediction
                k = self.exp_max
                alpha_i = torch.exp(-k * (1 - t))
                dalpha_i = k * alpha_i
                dlog_alpha_i = k * torch.ones_like(alpha_i)
                beta_i = 1 - alpha_i
                dbeta_i = -dalpha_i
                dlog_beta_i = dbeta_i / torch.clamp(beta_i, min=EPSILON)
            elif self.noise_type == "exponential_rev":
                # "x0" prediction
                k = self.exp_max
                beta_i = torch.exp(-k * t)
                dbeta_i = -k * beta_i
                dlog_beta_i = -k * torch.ones_like(beta_i)
                alpha_i = 1 - beta_i
                dalpha_i = -dbeta_i
                dlog_alpha_i = dalpha_i / torch.clamp(alpha_i, min=EPSILON)
            elif self.noise_type == "diffusion":
                t_rev = 1.0 - t
                beta_rate = (
                    self.beta_start + t_rev * (self.beta_end - self.beta_start)
                ) * self.T
                Gamma = (
                    self.beta_start * t_rev
                    + 0.5 * (self.beta_end - self.beta_start) * t_rev * t_rev
                ) * self.T
                alpha_i = torch.exp(-0.5 * Gamma)
                dalpha_i = 0.5 * beta_rate * alpha_i
                dlog_alpha_i = 0.5 * beta_rate
                beta_i = torch.sqrt(torch.clamp(1 - torch.exp(-Gamma), min=0.0))
                dbeta_i = (
                    -0.5
                    * torch.exp(-Gamma)
                    * beta_rate
                    / torch.clamp(beta_i, min=EPSILON)
                )
                dlog_beta_i = dbeta_i / torch.clamp(beta_i, min=EPSILON)
            else:
                raise ValueError(f"Unknown noise type: {self.noise_type}")
            alpha.append(torch.clamp(alpha_i, min=0.0, max=1.0))
            dalpha.append(dalpha_i)
            dlog_alpha.append(dlog_alpha_i)
            beta.append(torch.clamp(beta_i, min=0.0, max=1.0))
            dbeta.append(dbeta_i)
            dlog_beta.append(dlog_beta_i)
            if self.sigma_type == "zero":
                sigma_i = torch.zeros_like(t)
            elif self.sigma_type == "memoryless":
                if (
                    self.noise_type == "linear"
                    or self.noise_type == "exponential"
                    or self.noise_type == "exponential_rev"
                ):
                    sigma_i = self.sigma_scale * torch.sqrt(
                        torch.clamp(2 * dlog_alpha_i * beta_i, min=0.0)
                    )
                elif self.noise_type == "diffusion":
                    sigma_i = self.sigma_scale * torch.sqrt(
                        torch.clamp(2 * dlog_alpha_i, min=0.0)
                    )
                else:
                    sigma_i = self.sigma_scale * torch.sqrt(
                        torch.clamp(
                            2 * beta_i * (dlog_alpha_i * beta_i - dbeta_i), min=0.0
                        )
                    )
            sigma.append(sigma_i)
        return alpha, dalpha, beta, dbeta, sigma, dlog_alpha, dlog_beta

    def slice_input(self, x, input_start, input_end):
        """Inference-side: slice each sample's current state to the input window."""
        return [x[i][:, input_start[i] : input_end[i], ...] for i in range(len(x))]

    def prepare(self, x, device, valid_len, current_step):
        """Inference-side helper: current_step (int) → all schedule/noise/window
        coefficients sliced to their input/output windows, plus xt sliced to
        input window. Training-side stratified sampling does not go through
        this helper; it calls the lower-level scheduler methods directly.

        Returns dict with keys:
            time_schedules, time_schedules_derivative,
            alpha, dalpha, beta, dbeta, sigma, dlog_alpha, dlog_beta,
            input_start, input_end, output_start, output_end, xt
        """
        time_steps = self.get_time_steps(device, valid_len, current_step)
        time_schedules, time_schedules_derivative = self.get_time_schedules(
            device, valid_len, time_steps,
        )
        alpha, dalpha, beta, dbeta, sigma, dlog_alpha, dlog_beta = \
            self.get_noise_levels(device, valid_len, time_schedules)
        input_start, input_end, output_start, output_end = \
            self.get_windows(valid_len, time_steps)
        xt = self.slice_input(x, input_start, input_end)

        # Slice all coefficients to their respective windows
        batch_size = len(valid_len)
        time_schedules = [time_schedules[i][input_start[i]:input_end[i]] for i in range(batch_size)]
        time_schedules_derivative = [time_schedules_derivative[i][output_start[i]:output_end[i]] for i in range(batch_size)]
        alpha = [alpha[i][output_start[i]:output_end[i]] for i in range(batch_size)]
        dalpha = [dalpha[i][output_start[i]:output_end[i]] for i in range(batch_size)]
        beta = [beta[i][output_start[i]:output_end[i]] for i in range(batch_size)]
        dbeta = [dbeta[i][output_start[i]:output_end[i]] for i in range(batch_size)]
        sigma = [sigma[i][output_start[i]:output_end[i]] for i in range(batch_size)]
        dlog_alpha = [dlog_alpha[i][output_start[i]:output_end[i]] for i in range(batch_size)]
        dlog_beta = [dlog_beta[i][output_start[i]:output_end[i]] for i in range(batch_size)]

        return {
            "time_schedules": time_schedules,
            "time_schedules_derivative": time_schedules_derivative,
            "input_start": input_start,
            "input_end": input_end,
            "output_start": output_start,
            "output_end": output_end,
            "alpha": alpha,
            "dalpha": dalpha,
            "beta": beta,
            "dbeta": dbeta,
            "sigma": sigma,
            "dlog_alpha": dlog_alpha,
            "dlog_beta": dlog_beta,
            "xt": xt,
        }

    # --- Streaming support ---

    def get_committable(self, total_frames):
        """Given total accumulated conditions, return how many frames can be committed.
        Currently, we suppose steps % chunk_size == 0 for simplicity."""
        committable_length = max(0, total_frames - self.chunk_size + 1)
        committable_steps = total_frames * (self.steps // self.chunk_size)
        return committable_length, committable_steps

    def get_step_rollback(self, seq_len):
        """Get the step count to subtract when wrapping the buffer by seq_len.
        Corresponds to how many steps were consumed by seq_len frames."""
        steps = seq_len * (self.steps // self.chunk_size)
        return steps


class T5TextCrossModule(nn.Module):
    """Cross-attention module for T5 text conditioning."""

    def __init__(
        self,
        len=512,
        dim=4096,
        t5_size="xxl",
        checkpoint_path=None,
        tokenizer_path=None,
        drop_out=0.1,
        cross_rope=False,
        input_keys={
            "text": "text",
            "text_end": "text_end",
        },
        cache_encoded=True,
    ):
        assert checkpoint_path is not None and tokenizer_path is not None, (
            "T5 checkpoint and tokenizer paths must be provided."
        )
        super().__init__()
        self.len = len
        self.dim = dim
        self.cross_attn_norm = True
        self.cross_rope = cross_rope
        self.drop_out = drop_out
        self.input_keys = input_keys

        self.text_encoder = T5EncoderModel(
            text_len=len,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
            shard_fn=None,
            t5_size=t5_size,
        )
        # When False, text_cache is cleared after each encode/get_context call
        # so it doesn't accumulate across training steps (~190KB per caption,
        # OOM on 1M+ unique captions).
        self.cache_encoded = bool(cache_encoded)
        self.text_cache = {}

    def encode(self, text_list, device):
        """Encode text list with cache. Returns List[Tensor]."""
        # Deduplicate uncached texts
        texts_to_encode = []
        for text in text_list:
            if text not in self.text_cache and text not in texts_to_encode:
                texts_to_encode.append(text)

        # Batch encode deduplicated texts
        if texts_to_encode:
            self.text_encoder.model.to(device)
            encoded = self.text_encoder(texts_to_encode, device)
            for text, feature in zip(texts_to_encode, encoded):
                self.text_cache[text] = feature.cpu()

        # Collect from cache
        out = [self.text_cache[text].to(device) for text in text_list]
        if not self.cache_encoded:
            self.text_cache.clear()
        return out

    def get_context(self, x, valid_len, device, param_dtype, training=False):
        """Mask-based unified context.

        Both uniform (text=str) and multi-segment (text=List[str] + text_end)
        inputs are funneled into a single representation: for each sample,
        concatenate all (unique) segments along the K dimension and produce a
        (T_i, S_i) cross-attn mask telling which tokens each query frame may see.

        Returns:
            context: List[Tensor (S_i, D_kv)]   (S_i = num_seg_i * L_text)
            metadata: dict with
              'full_text': List[str] (logging only)
              'cross_attn_mask': List[BoolTensor (T_i, S_i)]
              'text_k_rope_ids' (cross_rope=True): List[Tuple]; f_ids
                  has shape (S_i,) and each segment's L_text tokens share the
                  segment-start-frame value. Null samples carry f_ids=0
                  throughout (single segment, L_text tokens).
              'text_q_rope_ids' (cross_rope=True): List[Tuple|None]
                  per-sample Q rope override; non-None only for null samples
                  (zeros over T_i) so cross-attn phase = 0 (matches K=0).
        """
        text_key = self.input_keys.get("text", "text")
        text_end_key = self.input_keys.get("text_end", "text_end")
        metadata = {}

        text_input = x[text_key] if text_key in x else ["" for _ in range(len(valid_len))]
        text_end_input = x.get(text_end_key, None)

        # Normalize each sample to (text_list, text_end_list, is_null).
        # text_end_list always starts with 0 and ends with valid_len[i].
        norm = []
        for i in range(len(valid_len)):
            is_null = training and np.random.rand() <= self.drop_out
            if is_null:
                norm.append(([""], [0, valid_len[i]], True))
                continue
            if isinstance(text_input[i], list):
                tl = text_input[i]
                assert text_end_input is not None, (
                    "text_end required when text is List[str] (multi-segment)"
                )
                te = [0] + [min(int(t), valid_len[i]) for t in text_end_input[i]]
            else:
                tl = [text_input[i]]
                te = [0, valid_len[i]]
            # Force last segment to extend to valid_len so batch-padding
            # frames (generate passes the batch-max seq_len as every sample's
            # generated_len) get text K coverage instead of an all-False mask.
            # Mirrors PositionCrossModule. No-op for the single-segment
            # branch where te[-1] is already valid_len[i].
            te[-1] = valid_len[i]
            norm.append((tl, te, False))

        # Encode all unique texts (cache-aware dedup)
        texts_to_encode = []
        for tl, _, _ in norm:
            for t in tl:
                if t not in self.text_cache and t not in texts_to_encode:
                    texts_to_encode.append(t)
        if texts_to_encode:
            self.text_encoder.model.to(device)
            encoded = self.text_encoder(texts_to_encode, device)
            for t, f in zip(texts_to_encode, encoded):
                self.text_cache[t] = f.cpu()

        all_context = []
        all_mask = []
        all_text_k_rope_ids = [] if self.cross_rope else None
        text_q_rope_override = [] if self.cross_rope else None
        full_text = []

        for i, (tl, te, is_null) in enumerate(norm):
            seg_tensors = [self.text_cache[t].to(device).to(param_dtype) for t in tl]
            # T5 returns variable-length tokens per caption (trimmed to its
            # actual mask), so segments may have different K-dim lengths.
            # Use cumulative offsets, NOT k * L_text (would mis-stride and
            # leak/lose K tokens whenever segments have different lengths).
            seg_K_lens = [t.size(0) for t in seg_tensors]
            seg_K_offsets = [0]
            for L in seg_K_lens:
                seg_K_offsets.append(seg_K_offsets[-1] + L)
            seg_lens = [t - b for t, b in zip(te[1:], te[:-1])]

            context_i = torch.cat(seg_tensors, dim=0)  # (sum(seg_K_lens), D_kv)
            S_i = context_i.size(0)
            all_context.append(context_i)

            full_text.append(
                " ////////// ".join([f"{u} //dur:{d}" for u, d in zip(tl, seg_lens)])
            )

            # Frame f belongs to segment k where te[k] <= f < te[k+1];
            # mark mask[f, seg_K_offsets[k]:seg_K_offsets[k+1]] = True.
            T_i = valid_len[i]
            mask = torch.zeros(T_i, S_i, dtype=torch.bool, device=device)
            for k, (s_f, e_f) in enumerate(zip(te[:-1], te[1:])):
                s_f = max(0, s_f)
                e_f = min(T_i, e_f)
                if s_f >= e_f:
                    continue
                mask[s_f:e_f, seg_K_offsets[k]:seg_K_offsets[k+1]] = True
            all_mask.append(mask)

            if self.cross_rope:
                # K rope: per-token segment-start-frame; null samples → 0.
                f_ids = torch.empty(S_i, dtype=torch.long, device=device)
                if is_null:
                    f_ids.zero_()
                else:
                    for k, s_f in enumerate(te[:-1]):
                        f_ids[seg_K_offsets[k]:seg_K_offsets[k+1]] = s_f
                all_text_k_rope_ids.append((f_ids, None, None))

                # Q rope override: real → None (use self-attn default arange);
                # null → zeros so phase = Q(0) - K(0) = 0 (rope-off equivalent).
                if is_null:
                    text_q_rope_override.append(
                        (torch.zeros(T_i, dtype=torch.long, device=device), None, None)
                    )
                else:
                    text_q_rope_override.append(None)

        metadata["full_text"] = full_text
        metadata["cross_attn_mask"] = all_mask
        if self.cross_rope:
            metadata["text_k_rope_ids"] = all_text_k_rope_ids
            metadata["text_q_rope_ids"] = text_q_rope_override
        if not self.cache_encoded:
            self.text_cache.clear()
        return all_context, metadata

    def get_null_context(self, real_context, device, param_dtype, valid_len):
        """Build null context with the mask-path convention.

        Each sample gets a single null segment of L_text tokens (encoded("")),
        an all-True (T_i, L_text) mask, K rope = 0, Q rope override = 0.
        This matches the original uniform-null math (phase = 0 = rope-off
        equivalent) and is the only configuration consistent with the mask
        path's unified shape.

        Args:
            real_context: List[Tensor], used to read batch size.
            valid_len: List[int], per-sample query frame count (T_i).

        Returns:
            null_context: List[Tensor (L_text, D_kv)]
            metadata: dict with cross_attn_mask + (when self.cross_rope=True)
                text_k_rope_ids and text_q_rope_ids.
        """
        batch_size = len(real_context)
        encoded_null = self.encode([""], device)[0].to(param_dtype)
        L_text = encoded_null.size(0)
        metadata = {}

        null_context = [encoded_null for _ in range(batch_size)]
        metadata["cross_attn_mask"] = [
            torch.ones(valid_len[i], L_text, dtype=torch.bool, device=device)
            for i in range(batch_size)
        ]

        if self.cross_rope:
            metadata["text_k_rope_ids"] = [
                (torch.zeros(L_text, dtype=torch.long, device=device), None, None)
                for _ in range(batch_size)
            ]
            metadata["text_q_rope_ids"] = [
                (torch.zeros(valid_len[i], dtype=torch.long, device=device), None, None)
                for i in range(batch_size)
            ]
        return null_context, metadata

    # --- Streaming state management (mask path) ---

    def init_stream(self, batch_size):
        # Per-sample list of segments; each entry is (text, start_frame_in_buffer).
        # End frame of segment k is start of segment k+1 (or stream_frames[i]).
        self.stream_segments = [[] for _ in range(batch_size)]
        # Total frames pushed via update_stream (one per call).
        self.stream_frames = [0 for _ in range(batch_size)]

    def update_stream(self, x, device, param_dtype):
        """Add one frame of conditioning. If text matches the latest segment's
        text, extend (implicit via stream_frames++); otherwise start a new
        segment whose start_frame = current frame index.

        Pre-encodes any uncached texts so get_stream_context is cache-only.
        """
        text_key = self.input_keys.get("text", "text")
        text_input = x[text_key]

        # Cache uncached texts (incl. "" so null path can pull from cache too).
        to_encode = []
        for t in list(text_input) + [""]:
            if t not in self.text_cache and t not in to_encode:
                to_encode.append(t)
        if to_encode:
            self.text_encoder.model.to(device)
            encoded = self.text_encoder(to_encode, device)
            for t, f in zip(to_encode, encoded):
                self.text_cache[t] = f.cpu()

        for i in range(len(self.stream_segments)):
            cur_text = text_input[i] if i < len(text_input) else ""
            cur_frame = self.stream_frames[i]
            if (
                len(self.stream_segments[i]) > 0
                and self.stream_segments[i][-1][0] == cur_text
            ):
                # Same as latest segment → frame counter handles extension.
                pass
            else:
                self.stream_segments[i].append((cur_text, cur_frame))
            self.stream_frames[i] += 1

    def get_stream_context(self, start_index, end_index, device, param_dtype):
        """Return mask-form context for buffer window [start_index, end_index).

        Returns:
            contexts: List[Tensor (S_i, D_kv)]
            metadata: {
                'cross_attn_mask': List[BoolTensor (window_len, S_i)]
                'text_k_rope_ids' (cross_rope=True): List[Tuple]
                    f_ids = segment_start_frame (buffer-absolute), repeated L_text
                    times within each segment.
                'text_q_rope_ids' (cross_rope=True): List[None] (real path).
            }
        """
        window_len = end_index - start_index
        contexts = []
        masks = []
        rope_ids = [] if self.cross_rope else None
        text_q_rope_overrides = [] if self.cross_rope else None

        for i in range(len(self.stream_segments)):
            segs = self.stream_segments[i]
            if len(segs) == 0:
                raise RuntimeError(
                    f"sample {i}: stream not initialized (call update_stream first)"
                )
            seg_starts = [s for _, s in segs]
            seg_ends = seg_starts[1:] + [self.stream_frames[i]]

            # Active = segments overlapping [start_index, end_index).
            active = []
            for k, (text, s_start) in enumerate(segs):
                s_end = seg_ends[k]
                if s_end <= start_index or s_start >= end_index:
                    continue
                active.append((text, s_start, s_end))
            if len(active) == 0:
                # Window entirely outside any segment — fall back to null seg
                # covering the whole window so attention still has valid K.
                active = [("", start_index, end_index)]

            seg_tensors = [
                self.text_cache[text].to(device).to(param_dtype)
                for (text, _, _) in active
            ]
            # T5 returns variable-length per-caption tokens; segments may
            # have different K-dim lengths. Use cumulative offsets, not
            # k * L_text (would mis-stride and leak/lose K tokens).
            seg_K_lens = [t.size(0) for t in seg_tensors]
            seg_K_offsets = [0]
            for L in seg_K_lens:
                seg_K_offsets.append(seg_K_offsets[-1] + L)
            context_i = torch.cat(seg_tensors, dim=0)
            S_i = context_i.size(0)
            contexts.append(context_i)

            mask = torch.zeros(window_len, S_i, dtype=torch.bool, device=device)
            for k, (_, s_start, s_end) in enumerate(active):
                local_start = max(0, s_start - start_index)
                local_end = min(window_len, s_end - start_index)
                if local_start < local_end:
                    mask[local_start:local_end, seg_K_offsets[k]:seg_K_offsets[k+1]] = True
            masks.append(mask)

            if self.cross_rope:
                f_ids = torch.empty(S_i, dtype=torch.long, device=device)
                for k, (_, s_start, _) in enumerate(active):
                    f_ids[seg_K_offsets[k]:seg_K_offsets[k+1]] = s_start
                rope_ids.append((f_ids, None, None))
                text_q_rope_overrides.append(None)

        metadata = {"cross_attn_mask": masks}
        if self.cross_rope:
            metadata["text_k_rope_ids"] = rope_ids
            metadata["text_q_rope_ids"] = text_q_rope_overrides
        return contexts, metadata

    def trim_stream(self, trim_len):
        """Shift segment starts back by trim_len; drop segments fully out of buffer.

        Negative starts are kept (clamped to ROPE_MIN_INDEX) so K rope can still
        encode pre-buffer positions via rope_params_with_negative.
        """
        for i in range(len(self.stream_segments)):
            segs = self.stream_segments[i]
            if len(segs) == 0:
                self.stream_frames[i] -= trim_len
                continue
            seg_starts = [s for _, s in segs]
            seg_ends = seg_starts[1:] + [self.stream_frames[i]]
            new_segs = []
            for k, (text, s_start) in enumerate(segs):
                s_end = seg_ends[k]
                new_start = s_start - trim_len
                new_end = s_end - trim_len
                if new_end <= 0:
                    continue  # fully before buffer
                new_segs.append((text, max(new_start, ROPE_MIN_INDEX)))
            self.stream_segments[i] = new_segs
            self.stream_frames[i] -= trim_len


class DiffForcingWanModel(nn.Module):
    def __init__(
        self,
        input_dim=256,
        mean_path=None,
        std_path=None,
        loss_W=None,
        loss_w_coefficient=1.0,
        hidden_dim=1024,
        ffn_dim=2048,
        freq_dim=256,
        num_heads=8,
        num_layers=8,
        time_embedding_scale=1.0,
        attn_type="full",     # "causal" | "full" | "partial"
        rope_channel_split=[1, 0, 0],
        spatial_shape=(1, 1),
        prediction_type="vel",  # "vel", "x0", "eps"
        text_config={
            "len": 512,
            "dim": 4096,
            "cross_rope": False,
        },
        schedule_config={
            "noise_type": "linear",
            "chunk_size": 5,
            "steps": 10,
            "random_epsilon": 0.00,
        },
        cfg_config={
            "text_scale": 5.0,
            "null_scale": -4.0,
        },
        input_keys={
            "feature": "feature",
            "feature_length": "feature_length",
            "text": "text",
            "text_end": "text_end",
        },
        model_input_dim=None,
    ):
        super().__init__()
        self.input_keys = input_keys

        self.mean_path = mean_path
        self.std_path = std_path
        self.input_dim = input_dim
        loss_matrix = load_quadratic_loss_matrix(
            loss_W, loss_w_coefficient, self.input_dim
        )
        if loss_matrix is None:
            self.loss_matrix = None
        else:
            self.register_buffer("loss_matrix", loss_matrix)
        # The diffusion state/output dimensionality remains ``input_dim``.
        # Conditioned variants may expose extra clean channels only to the
        # denoiser (packed root + latent), without diffusing or
        # scoring those channels.
        self.model_input_dim = (
            self.input_dim if model_input_dim is None else int(model_input_dim)
        )
        self.spatial_shape = tuple(spatial_shape)
        self.spatial_size = math.prod(self.spatial_shape)
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.time_embedding_scale = time_embedding_scale
        assert attn_type in ("causal", "full", "partial"), (
            f"attn_type must be one of 'causal'/'full'/'partial', got {attn_type!r}"
        )
        self.attn_type = attn_type
        self.rope_channel_split = rope_channel_split
        self.prediction_type = prediction_type
        self.cfg_config = cfg_config
        self.schedule_config = schedule_config
        # The multi-window mask is only sound for attn_type='partial': it
        # forces clean x clean to be causal and W_k x clean to be a strict
        # prefix, which contradicts full's "no restriction" and causal's
        # "global sequence causal". Force n_windows=1 under these attn_types
        # so the packed sequence degenerates to a single window and the
        # multi-window mask is bypassed in forward.
        if self.attn_type in ("full", "causal"):
            cfg_n = int(self.schedule_config.get("train_n_windows", 10))
            if cfg_n != 1:
                import warnings
                warnings.warn(
                    f"attn_type={self.attn_type!r} is incompatible with "
                    f"multi-window training (multi-window mask differs from "
                    f"full/causal inference mask). "
                    f"Forcing train_n_windows from {cfg_n} to 1.",
                    stacklevel=2,
                )
                self.schedule_config["train_n_windows"] = 1
        self.time_scheduler = TriangularTimeScheduler(schedule_config)
        # Cross-attention module (text)
        self.text_module = T5TextCrossModule(**text_config)

        mean, std = load_statistics(self.mean_path, self.std_path, input_dim)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        self.model = WanModel(
            patch_size=(1, 1, 1),
            text_len=self.text_module.len,
            text_dim=self.text_module.dim,
            cross_attn_norm=self.text_module.cross_attn_norm,
            cross_rope=self.text_module.cross_rope,
            in_dim=self.model_input_dim,
            dim=self.hidden_dim,
            ffn_dim=self.ffn_dim,
            freq_dim=self.freq_dim,
            out_dim=self.input_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            window_size=(-1, -1),
            qk_norm=True,
            eps=1e-6,
            causal=(self.attn_type == "causal"),
            rope_channel_split=self.rope_channel_split,
        )
        self.param_dtype = torch.float32

    # ------------------------------------------------------------------
    # Conditioning extension points
    # ------------------------------------------------------------------
    # The base model deliberately keeps these hooks as no-ops.  Position
    # conditioning lives in models/diffusion_forcing_position_wan.py while
    # this class remains bit-for-bit compatible for existing configs.

    def _sample_training_noise(self, x0, inputs, batch_index):
        return torch.randn_like(x0)

    def _get_diffusion_state(self, inputs):
        return inputs["feature"]

    def _prepare_training_condition(
        self, inputs, normalized_feature, valid_len, device
    ):
        return None

    def _condition_training_input(
        self, noisy, clean, condition, batch_index, start, end
    ):
        return noisy

    def _prepare_generation_condition(self, inputs, valid_len, seq_len, device):
        return None

    def _condition_generation_input(self, noisy, condition, starts, ends):
        return noisy

    def _apply_generation_state_condition(self, generated, condition):
        return generated

    def _finalize_generation_output(self, generated, condition, valid_len):
        return generated

    def _generation_output_metadata(self, condition, valid_len):
        return {}

    def _init_stream_condition(
        self, batch_size, buf_len, device, history_position=None,
        history_latent=None,
    ):
        return None

    def _update_stream_condition(self, inputs, device):
        return None

    def _commit_stream_condition(self, index):
        return None

    def _rollback_stream_condition(self, trim_len):
        return None

    def _condition_stream_input(self, noisy, starts, ends):
        return noisy

    def _apply_stream_state_condition(self):
        return None

    def _finalize_stream_output(self, generated, start, end):
        return generated

    def _stream_output_metadata(self, start, end):
        return {}

    def _extract_inputs(self, x):
        """Extract inputs from x using input_keys mapping."""
        inputs = {}
        for internal_key, external_key in self.input_keys.items():
            if external_key in x:
                inputs[internal_key] = x[external_key]
        return inputs

    def preprocess(self, x):
        """Convert last-channel format to channel-first, padding to 4D (C, T, H, W).
        (T, C) -> (C, T, 1, 1)
        (T, H, C) -> (C, T, H, 1)
        (T, H, W, C) -> (C, T, H, W)
        """
        for i in range(len(x)):
            ndim = x[i].ndim
            if ndim == 2:  # (T, C)
                x[i] = x[i].permute(1, 0)[:, :, None, None]
            elif ndim == 3:  # (T, H, C)
                x[i] = x[i].permute(2, 0, 1)[:, :, :, None]
            elif ndim == 4:  # (T, H, W, C)
                x[i] = x[i].permute(3, 0, 1, 2)
        return x

    def postprocess(self, x):
        """Reverse of preprocess: channel-first 4D back to last-channel, stripping padding dims.
        (C, T, 1, 1) -> (T, C)
        (C, T, H, 1) -> (T, H, C)
        (C, T, H, W) -> (T, H, W, C)
        """
        for i in range(len(x)):
            shape = x[i].shape  # (C, T, H, W)
            if shape[2] == 1 and shape[3] == 1:  # (C, T, 1, 1) -> (T, C)
                x[i] = x[i][:, :, 0, 0].permute(1, 0)
            elif shape[3] == 1:  # (C, T, H, 1) -> (T, H, C)
                x[i] = x[i][:, :, :, 0].permute(1, 2, 0)
            else:  # (C, T, H, W) -> (T, H, W, C)
                x[i] = x[i].permute(1, 2, 3, 0)
        return x

    @staticmethod
    def _shift_rope_ids(rope_ids, starts):
        """Subtract starts[i] from f_ids of each tuple, making K rope window-relative
        so cross-attn phase matches Q's default-arange convention even when input
        window doesn't start at frame 0 (content_len truncation).
        Handles per-frame (List[List[Tuple]]), uniform (List[Tuple]), or per-sample
        optional (List[Tuple|None]). None per-sample entries pass through.
        Tuples with f_ids=None (null/CFG placeholders) are also passed through.
        """
        if rope_ids is None:
            return None

        def shift_tuple(tup, s):
            f, h, w = tup
            return ((f - s) if f is not None else None, h, w)

        result = []
        for sample_ids, s in zip(rope_ids, starts):
            if sample_ids is None:
                result.append(None)
            elif isinstance(sample_ids, list):
                result.append([shift_tuple(t, s) for t in sample_ids])
            else:
                result.append(shift_tuple(sample_ids, s))
        return result

    def _patch_seq_len(self, frame_seq_len):
        """Frame-count seq_len → patch-count seq_len."""
        return frame_seq_len * self.spatial_size

    def _expand_schedule_to_patch(self, time_schedules_input):
        """Per-frame schedule → per-patch by repeating each frame value spatial_size times.
        No-op when spatial_size == 1.
        """
        if self.spatial_size == 1:
            return time_schedules_input
        return [s.repeat_interleave(self.spatial_size) for s in time_schedules_input]

    def _expand_context_to_patch(self, context):
        """Per-frame nested context → per-patch by repeating each frame's context spatial_size times.
        Uniform per-sample context (List[Tensor]) is returned unchanged.
        No-op when spatial_size == 1.
        """
        if self.spatial_size == 1:
            return context
        if len(context) > 0 and isinstance(context[0], (list, tuple)):
            return [
                [u for u in sample_ctx for _ in range(self.spatial_size)]
                for sample_ctx in context
            ]
        return context

    def _expand_mask_to_patch(self, masks):
        """Cross-attn mask Q dim (frames) → patch dim by repeating each row
        spatial_size times. K dim untouched. No-op when spatial_size == 1
        or masks is None.
        """
        if masks is None or self.spatial_size == 1:
            return masks
        return [m.repeat_interleave(self.spatial_size, dim=0) for m in masks]

    def _build_partial_mask(self, input_start, input_end, output_start, output_end, device):
        """Per-sample partial-causal attention mask (used when attn_type='partial').

        Layout per sample:
          [input_start, output_start) → history: causal sliding window of size history_len
          [output_start, output_end)  → target : bidirectional + sees all history
        When spatial > 1, repeat-interleave to patch level (F·H·W, F·H·W).
        """
        spatial = self.spatial_size
        masks = []
        for is_, ie_, os_, _oe in zip(input_start, input_end, output_start, output_end):
            L_frame = ie_ - is_
            history_len = os_ - is_
            sliding_window = history_len
            m = torch.zeros(L_frame, L_frame, dtype=torch.bool, device=device)
            for j in range(history_len):
                m[j, max(0, j - sliding_window + 1):j + 1] = True
            m[history_len:, :history_len] = True   # target -> history (fully visible)
            m[history_len:, history_len:] = True   # target x target (bidirectional)
            if spatial > 1:
                m = m.repeat_interleave(spatial, dim=0).repeat_interleave(spatial, dim=1)
            masks.append(m)
        return masks

    def _build_multi_window_attn_mask(self, clean_len, windows, clean_frame_start, device):
        """Self-attn mask for multi-window packed training.

        Packed sequence layout per sample: [clean (clean_len tokens)] + concat over k
        of W_k tokens, where W_k = frames [windows[k][0], windows[k][1]).
        The clean section corresponds to frames
        [clean_frame_start, clean_frame_start + clean_len).
        - clean × clean: causal (clean[i] sees clean[0..i]).
        - clean × W: blocked (clean doesn't depend on noised tokens).
        - W_k × clean[0:s_k - clean_frame_start]: visible — W_k sees every
          clean token in the packed sequence whose absolute frame index is
          less than s_k (strict causal prefix in frame coords).
        - W_k × W_k: bidirectional (denoising chunk full attention).
        - W_k × W_j, j != k: blocked (independent denoising instances).
        When spatial > 1, repeat-interleave to patch level
        (L_frame·spatial, L_frame·spatial).
        """
        sizes = [e - s for s, e in windows]
        total_len = clean_len + sum(sizes)
        mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=device)
        mask[:clean_len, :clean_len] = torch.tril(
            torch.ones(clean_len, clean_len, dtype=torch.bool, device=device)
        )
        cursor = clean_len
        for s, e in windows:
            L = e - s
            s_packed = s - clean_frame_start    # packed col for frame s
            if s_packed > 0:
                mask[cursor:cursor + L, :s_packed] = True
            mask[cursor:cursor + L, cursor:cursor + L] = True
            cursor += L
        if self.spatial_size > 1:
            mask = mask.repeat_interleave(
                self.spatial_size, dim=0
            ).repeat_interleave(self.spatial_size, dim=1)
        return mask

    @staticmethod
    def _gather_for_packed(per_frame_tensor, clean_frame_start, clean_len, windows):
        """Gather rows from a per-frame tensor (T_i, ...) into the packed
        layout [clean section | W_0 | W_1 | ...]. The clean section uses
        rows [clean_frame_start, clean_frame_start + clean_len); each W_k
        uses rows [s_k, e_k). Returns a tensor of shape (packed_len, ...)
        with packed_len = clean_len + sum(L_k).
        """
        parts = []
        if clean_len > 0:
            parts.append(per_frame_tensor[clean_frame_start:clean_frame_start + clean_len])
        for s, e in windows:
            parts.append(per_frame_tensor[s:e])
        return torch.cat(parts, dim=0)

    def _gather_text_q_rope_packed(self, q_rope_ids_full,
                                   clean_frame_starts, clean_lens, all_windows):
        """Gather per-sample text_q_rope_ids tuples (per-sample List[Tuple|None])
        to the packed layout. Per-sample None entries pass through; non-None
        tuples have their f_ids (length T_i) gathered to packed_len following
        the packed layout. h_ids / w_ids are forwarded unchanged.
        """
        if q_rope_ids_full is None:
            return None
        result = []
        for i, item in enumerate(q_rope_ids_full):
            if item is None:
                result.append(None)
                continue
            f_ids, h_ids, w_ids = item
            if f_ids is None:
                result.append(item)
                continue
            gathered = self._gather_for_packed(
                f_ids, clean_frame_starts[i], clean_lens[i], all_windows[i],
            )
            result.append((gathered, h_ids, w_ids))
        return result

    def _build_multi_window_rope_ids(self, clean_len, windows, clean_frame_start, device):
        """Self-attn rope f_ids: per-token position **relative to clean_frame_start**.

        Clean: arange(0, clean_len)。
        W_k: arange(s_k - clean_frame_start, e_k - clean_frame_start)。
        Using positions relative to clean_frame_start matches the inference
        path's default arange(0, e) convention. The Q-K rope phase is still
        the frame difference (the constant offset cancels out after
        subtraction), so the semantics are unchanged.
        """
        parts = [torch.arange(clean_len, device=device, dtype=torch.long)]
        for s, e in windows:
            parts.append(torch.arange(
                s - clean_frame_start, e - clean_frame_start,
                device=device, dtype=torch.long,
            ))
        return (torch.cat(parts), None, None)

    def forward(self, x):
        """Multi-window packed training.

        Per sample (valid_len = T_i), divide t-range [0, (T_i+c-1)/c) into
        `train_n_windows` equal buckets (default 10, configurable via
        schedule_config); each bucket yields one uniform t_k. Windows are
        derived from t_k via `time_scheduler.get_windows`, so the choice of
        n_windows stays consistent with the inference path's formula and
        training cost is independent of sequence length.
        The packed self-attn sequence is
            [clean feature | W_0 | W_1 | ... | W_{N-1}]
        wired together by `_build_multi_window_attn_mask` and per-token rope
        ids from `_build_multi_window_rope_ids`. Cross-attn is uniform (full
        True mask, no rope override, no cross-rope). Loss is per-token MSE
        averaged across all window tokens in the batch.

        When spatial > 1, the metadata (time / attn_mask / cross_attn_mask /
        seq_len) is expanded to patch granularity; rope_ids' f_ids stays at
        frame level since rope_apply broadcasts across H/W internally.
        """
        x = self._extract_inputs(x)
        feature_original = self._get_diffusion_state(x)
        feature_length = x["feature_length"]
        feature_original = (feature_original - self.mean) / self.std
        batch_size = feature_original.shape[0]
        pad_seq_len = feature_original.shape[1]
        device = feature_original.device
        n_windows = int(self.schedule_config.get("train_n_windows", 10))

        valid_len = []
        feature = []
        for i in range(batch_size):
            length = min(feature_length[i].item(), pad_seq_len)
            valid_len.append(length)
            feature.append(feature_original[i, :length, ...])
        feature = self.preprocess(feature)  # list of (C, T_i, 1, 1)
        training_condition = self._prepare_training_condition(
            x, feature_original, valid_len, device,
        )

        # Text context + metadata. cross_attn_mask handles multi-segment Q->K
        # visibility; text_k_rope_ids / text_q_rope_ids carry per-segment rope
        # info when cross_rope=True (None otherwise).
        context, ctx_meta = self.text_module.get_context(
            x, valid_len, device, self.param_dtype, training=True,
        )
        cross_attn_mask_full = ctx_meta["cross_attn_mask"]
        text_k_rope_ids_full = ctx_meta.get("text_k_rope_ids")
        text_q_rope_ids_full = ctx_meta.get("text_q_rope_ids")

        all_xt = []
        all_t = []
        all_rope = []
        all_attn_mask = []
        all_cross_attn_mask = []
        all_targets = []  # per-sample list of (target_tensor, seq_start, seq_end)
        # Per-sample packed layout (used after the loop to gather Q-side text rope
        # and to shift K-side rope into the packed coordinate frame).
        all_clean_frame_starts = []
        all_clean_lens = []
        all_packed_windows = []

        for i in range(batch_size):
            T_i = valid_len[i]
            x0_i = feature[i]
            eps_i = self._sample_training_noise(x0_i, x, i)

            # Split t-range [0, t_max) into n_windows equal buckets; one
            # uniform sample per bucket. Bucket size scales with t_max so the
            # window count stays constant across sequence lengths.
            t_max = self.time_scheduler.get_t_max(T_i)
            bucket = t_max / n_windows
            t_values = [
                torch.tensor(
                    np.random.uniform(k * bucket, (k + 1) * bucket),
                    device=device,
                )
                for k in range(n_windows)
            ]
            input_start_all, _, output_start, output_end = self.time_scheduler.get_windows(
                [T_i] * n_windows, t_values, training=True,
            )
            windows = list(zip(output_start, output_end))

            # t_values is non-decreasing => window starts are
            # monotonically non-decreasing => clean section bounds:
            #   - clean_frame_start = min(input_start): the earliest window's
            #     history start
            #   - clean_frame_end   = max(output_start) = windows[-1][0]
            # Frames beyond clean_frame_end are not read by any W_k, and
            # frames before clean_frame_start are never visible to any W_k
            # (mask gates by frame index), so trimming the buffer to this
            # span is sufficient.
            clean_frame_start = min(input_start_all)
            clean_frame_end = windows[-1][0]
            clean_len = clean_frame_end - clean_frame_start

            # N schedules over T_i frames, one per t_k. Reuses existing
            # scheduler so noise_type / random_epsilon stay consistent with
            # generate.
            schedules, _ = self.time_scheduler.get_time_schedules(
                device, [T_i] * n_windows, t_values, training=True,
            )
            alphas, dalphas, betas, dbetas, _sg, _dla, _dlb = (
                self.time_scheduler.get_noise_levels(
                    device, [T_i] * n_windows, schedules,
                )
            )
            # Clean section: same α·x0 + β·noise noising path as every window,
            # so the latent agrees with the (possibly random_epsilon-jittered)
            # schedule that's also handed to the model as time embedding.
            clean_schedule, _ = self.time_scheduler.get_time_schedules(
                device, [T_i], [torch.tensor(t_max, device=device)],
                training=True,
            )
            clean_alphas, _, clean_betas, _, _, _, _ = (
                self.time_scheduler.get_noise_levels(
                    device, [T_i], clean_schedule,
                )
            )

            # Assemble packed sequence [clean | W_0 | ... | W_{N-1}].
            ca = clean_alphas[0][clean_frame_start:clean_frame_end]
            cb = clean_betas[0][clean_frame_start:clean_frame_end]
            x0_c = x0_i[:, clean_frame_start:clean_frame_end, ...]
            eps_c = eps_i[:, clean_frame_start:clean_frame_end, ...]
            xt_c = (
                x0_c * ca[None, :, None, None]
                + eps_c * cb[None, :, None, None]
            )
            xt_c = self._condition_training_input(
                xt_c,
                x0_c,
                training_condition,
                i,
                clean_frame_start,
                clean_frame_end,
            )
            parts_xt = [xt_c]
            parts_time = [clean_schedule[0][clean_frame_start:clean_frame_end]]
            targets = []
            cursor = clean_len
            for k in range(n_windows):
                s, e = windows[k]
                L = e - s
                a = alphas[k][s:e]
                b = betas[k][s:e]
                x0_w = x0_i[:, s:e, ...]
                eps_w = eps_i[:, s:e, ...]
                xt_w = (
                    x0_w * a[None, :, None, None]
                    + eps_w * b[None, :, None, None]
                )
                xt_w = self._condition_training_input(
                    xt_w, x0_w, training_condition, i, s, e,
                )
                parts_xt.append(xt_w)
                parts_time.append(schedules[k][s:e])

                if self.prediction_type == "vel":
                    da = dalphas[k][s:e]
                    db = dbetas[k][s:e]
                    tgt = (
                        x0_w * da[None, :, None, None]
                        + eps_w * db[None, :, None, None]
                    )
                elif self.prediction_type == "x0":
                    tgt = x0_w
                elif self.prediction_type == "eps":
                    tgt = eps_w
                else:
                    raise ValueError(
                        f"Unknown prediction_type: {self.prediction_type!r}"
                    )
                targets.append((tgt, cursor, cursor + L))
                cursor += L

            xt_i = torch.cat(parts_xt, dim=1)
            assert xt_i.size(0) == self.model_input_dim, (
                f"conditioned denoiser input has {xt_i.size(0)} channels; "
                f"expected model_input_dim={self.model_input_dim}"
            )
            time_i = torch.cat(parts_time) * self.time_embedding_scale

            all_xt.append(xt_i)
            all_t.append(time_i)
            all_rope.append(
                self._build_multi_window_rope_ids(
                    clean_len, windows, clean_frame_start, device,
                )
            )
            all_attn_mask.append(
                self._build_multi_window_attn_mask(
                    clean_len, windows, clean_frame_start, device,
                )
            )
            # Cross-attn mask: gather per-frame rows from cross_attn_mask_full
            # to follow the packed [clean | W_k] layout. Frame-level here;
            # spatial expansion happens after the loop.
            all_cross_attn_mask.append(
                self._gather_for_packed(
                    cross_attn_mask_full[i], clean_frame_start, clean_len, windows,
                )
            )
            all_targets.append(targets)
            all_clean_frame_starts.append(clean_frame_start)
            all_clean_lens.append(clean_len)
            all_packed_windows.append(windows)

        # Expand frame-level time / cross_attn_mask to patch-level
        # (no-ops when spatial_size == 1).
        all_t = self._expand_schedule_to_patch(all_t)
        all_cross_attn_mask = self._expand_mask_to_patch(all_cross_attn_mask)
        # seq_len = max(pad_seq_len, max_packed). In n=1 mode
        # packed_length <= T_i <= pad_seq_len so pad_seq_len dominates and
        # the model sees the same padded geometry as the inference path.
        # With n>1 (partial only) packed can exceed pad_seq_len; the max
        # keeps the upper bound correct.
        seq_len_for_model = self._patch_seq_len(
            max(pad_seq_len, max(xt.size(1) for xt in all_xt))
        )
        text_seq_len = max(int(ctx.size(0)) for ctx in context)

        # Cross-attn rope (only active when cross_rope=True): K rope encodes
        # per-token segment-start-frame in absolute coords, Q rope override
        # exists for null samples (zeros over T_i). Both must be expressed in
        # the same packed coordinate frame as self-attn rope_ids (which is
        # relative to clean_frame_start), so:
        #   - K rope: shift by clean_frame_start (subtract) per sample
        #   - Q rope: gather per-frame f_ids to packed layout, then shift
        text_k_rope_ids = self._shift_rope_ids(
            text_k_rope_ids_full, all_clean_frame_starts,
        )
        text_q_rope_ids = self._gather_text_q_rope_packed(
            text_q_rope_ids_full, all_clean_frame_starts, all_clean_lens,
            all_packed_windows,
        )
        text_q_rope_ids = self._shift_rope_ids(
            text_q_rope_ids, all_clean_frame_starts,
        )

        # The multi-window mask is only meaningful for attn_type='partial'.
        # Under full/causal (where __init__ already forced n_windows=1), the
        # packed sequence is a single window, so we let WanModel apply
        # default full attention or its internal causal mask.
        final_attn_mask = all_attn_mask if self.attn_type == "partial" else None
        predicted_result = self.model(
            all_xt,
            all_t,
            context,
            seq_len_for_model,
            attn_mask=final_attn_mask,
            text_k_rope_ids=text_k_rope_ids,
            text_q_rope_ids=text_q_rope_ids,
            cross_attn_mask=all_cross_attn_mask,
            text_seq_len=text_seq_len,
            y=None,
            rope_ids=all_rope,
        )

        # Per-sample mean -> batch mean. Within each sample sum SE across
        # all W_k tokens and divide by the token count to get per-sample
        # MSE, then average across the batch.
        loss = 0.0
        for i in range(batch_size):
            pred_i = predicted_result[i]
            sample_se_sum = pred_i.new_zeros(())
            sample_n = 0
            for tgt, s, e in all_targets[i]:
                error = pred_i[:, s:e, ...] - tgt
                if self.loss_matrix is None:
                    sample_se_sum = sample_se_sum + (error ** 2).sum()
                else:
                    sample_se_sum = sample_se_sum + quadratic_error_sum(
                        error, self.loss_matrix
                    )
                sample_n += error.numel()
            loss = loss + sample_se_sum / max(sample_n, 1)
        loss = loss / batch_size
        return {"total": loss, "mse": loss}

    def generate(self, x):
        """
        Generation - Diffusion Forcing inference
        Uses triangular noise schedule, progressively generating from left to right

        Generation process:
        1. Start from t=0, gradually increase t
        2. Each t corresponds to a noise schedule: clean on left, noisy on right, gradient in middle
        3. After each denoising step, t increases slightly and continues
        """
        x = self._extract_inputs(x)
        feature_length = x["feature_length"]  # (B,)
        batch_size = len(feature_length)
        # extra_len pads the rollout with disposable frames so real frames
        # finish under full-width windows; 0 relies on the tail ramp-down.
        extra_len = self.schedule_config.get("extra_len", 0)
        seq_len = max(feature_length).item() + extra_len
        if (not getattr(self.time_scheduler, "t_max_tail", True)
                and extra_len < self.time_scheduler.chunk_size - 1):
            import warnings
            warnings.warn(
                f"t_max_tail=False with extra_len={extra_len} < chunk_size-1="
                f"{self.time_scheduler.chunk_size - 1}: the last frames of "
                f"every generated sequence will not finish denoising",
                stacklevel=2,
            )
        device = next(self.parameters()).device
        valid_len = []
        for i in range(batch_size):
            length = min(feature_length[i].item(), seq_len)
            valid_len.append(length)
        generated_len = [seq_len for _ in range(batch_size)]
        generation_condition = self._prepare_generation_condition(
            x, valid_len, seq_len, device,
        )

        # Initialize entire sequence as pure noise
        generated = torch.randn(
            batch_size, seq_len, *self.spatial_shape, self.input_dim, device=device
        )
        generated = [generated[i] for i in range(batch_size)]
        generated = self.preprocess(generated)
        self._apply_generation_state_condition(generated, generation_condition)

        # Precompute real text context (null built per-step from window-sliced real).
        text_context, metadata = self.text_module.get_context(
            x,
            generated_len,
            device,
            self.param_dtype,
            training=False,
        )
        full_text = metadata["full_text"]
        cross_attn_mask_full = metadata["cross_attn_mask"]
        text_k_rope_ids_full = metadata.get("text_k_rope_ids")

        total_steps = self.time_scheduler.get_total_steps(seq_len)
        # Progressively advance from t=0 to t=max_t
        for step in range(total_steps):
            s = self.time_scheduler.prepare(
                generated, device, generated_len, current_step=step,
            )
            time_schedules = s["time_schedules"]
            time_schedules_derivative = s["time_schedules_derivative"]
            alpha = s["alpha"]
            dalpha = s["dalpha"]
            beta = s["beta"]
            dbeta = s["dbeta"]
            sigma = s["sigma"]
            dlog_alpha = s["dlog_alpha"]
            dlog_beta = s["dlog_beta"]
            input_start_index = s["input_start"]
            input_end_index = s["input_end"]
            output_start_index = s["output_start"]
            output_end_index = s["output_end"]
            xt = s["xt"]
            xt = self._condition_generation_input(
                xt, generation_condition, input_start_index, input_end_index,
            )

            # time_schedules already sliced to input window by prepare()
            time_schedules_input = [
                time_schedules[i] * self.time_embedding_scale
                for i in range(batch_size)
            ]

            # Slice cross_attn_mask Q dim to input window (K untouched).
            window_cross_attn_mask = [
                cross_attn_mask_full[i][input_start_index[i]:input_end_index[i], :]
                for i in range(batch_size)
            ]
            # Shift K rope to window-relative coords (lockstep with forward path).
            window_text_k_rope = self._shift_rope_ids(text_k_rope_ids_full, input_start_index)

            # CFG: text_scale * pred_text + null_scale * pred_null
            ts_patch = self._expand_schedule_to_patch(time_schedules_input)
            patch_seq_len = self._patch_seq_len(seq_len)
            attn_mask = (
                self._build_partial_mask(input_start_index, input_end_index,
                                         output_start_index, output_end_index, device)
                if self.attn_type == "partial" else None
            )
            # Build null context matching window-sliced real shape.
            window_lens = [
                input_end_index[i] - input_start_index[i] for i in range(batch_size)
            ]
            window_null_context, window_null_meta = self.text_module.get_null_context(
                text_context, device, self.param_dtype, valid_len=window_lens,
            )
            window_null_mask = window_null_meta["cross_attn_mask"]
            window_null_text_k_rope = window_null_meta.get("text_k_rope_ids")
            window_null_text_q_rope = window_null_meta.get("text_q_rope_ids")

            # Doubled batch: [text..., null...]. One model.forward, split outputs.
            combined_ctx = self._expand_context_to_patch(text_context + window_null_context)
            combined_cross_attn_mask = self._expand_mask_to_patch(
                window_cross_attn_mask + window_null_mask
            )
            if window_text_k_rope is not None and window_null_text_k_rope is not None:
                combined_text_k_rope = window_text_k_rope + window_null_text_k_rope
            else:
                combined_text_k_rope = None
            # Real branch has no Q-override (cross falls back to self_attn rope);
            # null branch emits one when cross_rope=True. Concat [None]*B for real.
            if window_null_text_q_rope is not None:
                combined_text_q_rope = [None] * batch_size + list(window_null_text_q_rope)
            else:
                combined_text_q_rope = None
            combined_mask = attn_mask + attn_mask if attn_mask is not None else None

            combined_text_seq_len = max(
                int(c.size(0)) for c in (text_context + window_null_context)
            )
            pred_all = self.model(
                xt + xt,
                ts_patch + ts_patch,
                combined_ctx,
                patch_seq_len,
                attn_mask=combined_mask,
                text_k_rope_ids=combined_text_k_rope,
                text_q_rope_ids=combined_text_q_rope,
                cross_attn_mask=combined_cross_attn_mask,
                text_seq_len=combined_text_seq_len,
                y=None,
            )
            pred_text = pred_all[:batch_size]
            pred_null = pred_all[batch_size:]
            predicted_result = [
                self.cfg_config["text_scale"] * pt + self.cfg_config["null_scale"] * pn
                for pt, pn in zip(pred_text, pred_null)
            ]

            # All noise coefficients already sliced to output window by prepare()
            for i in range(batch_size):
                os, oe = output_start_index[i], output_end_index[i]
                pred_os = os - input_start_index[i]
                pred_oe = oe - input_start_index[i]
                predicted_result_i = predicted_result[i][:, pred_os:pred_oe, ...]
                generated_i = generated[i][:, os:oe, ...]
                dt = time_schedules_derivative[i][None, :, None, None]
                alpha_i = alpha[i][None, :, None, None]
                dalpha_i = dalpha[i][None, :, None, None]
                beta_i = beta[i][None, :, None, None]
                dbeta_i = dbeta[i][None, :, None, None]
                sigma_i = sigma[i][None, :, None, None]
                dlog_alpha_i = dlog_alpha[i][None, :, None, None]
                dlog_beta_i = dlog_beta[i][None, :, None, None]
                if self.prediction_type == "vel":
                    vel = predicted_result_i
                elif self.prediction_type == "x0":
                    vel = (
                        predicted_result_i * (-dlog_beta_i * alpha_i + dalpha_i)
                        + generated_i * dlog_beta_i
                    )
                elif self.prediction_type == "eps":
                    vel = (
                        predicted_result_i * (-dlog_alpha_i * beta_i + dbeta_i)
                        + generated_i * dlog_alpha_i
                    )
                st = (vel - generated_i * dlog_alpha_i) / (
                    (beta_i * dlog_alpha_i - dbeta_i) * beta_i
                )
                generated[i][:, os:oe, ...] += (
                    vel * dt
                    + st * 0.5 * sigma_i**2 * dt
                    + sigma_i * torch.sqrt(dt) * torch.randn_like(generated_i)
                )
            self._apply_generation_state_condition(generated, generation_condition)

        generated = self.postprocess(generated)  # list of (T, C)
        y_hat_out = []
        for i in range(batch_size):
            single_generated = generated[i][: valid_len[i], :] * self.std + self.mean
            y_hat_out.append(single_generated)
        y_hat_out = self._finalize_generation_output(
            y_hat_out, generation_condition, valid_len,
        )
        out = {}
        out["generated"] = y_hat_out
        out["text"] = full_text
        out.update(self._generation_output_metadata(generation_condition, valid_len))

        return out

    def init_generated(
        self,
        seq_len,
        batch_size=1,
        schedule_config={},
        history_latent=None,
        history_position=None,
    ):
        """Initialize streaming generation state.

        Args:
            seq_len: Model window size (how many frames WanModel processes per step).
            schedule_config: Optional schedule config overrides.
            history_latent: optional list of (h, ...) un-normalized latent tensors per sample
                            (typically vae.encode(motion) output). If provided:
                              - normalize via self.mean/self.std
                              - apply triangular noise matching the schedule at t=1
                                (ts[i] = clamp(1 - i/chunk_size, 0, 1) for i in [0, h))
                              - inject into buffer[:, :h, ...]
                              - set current_step = steps so wavefront starts just past history
                            If None: cold start (buffer all noise, current_step=0).

        Buffer is 2*seq_len. Model window is always buffer[0:seq_len].
        When conditions overflow seq_len, shift buffer by seq_len and restart.
        """
        self.schedule_config.update(schedule_config)
        content_len = self.schedule_config.get("content_len", None)
        if content_len is None:
            self.schedule_config["content_len"] = seq_len
        else:
            self.schedule_config["content_len"] = min(seq_len, content_len)
        self.time_scheduler = TriangularTimeScheduler(self.schedule_config)

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.buf_len = seq_len * 2
        self.current_step = 0
        self.current_commit = 0
        self.condition_frames = 0

        device = next(self.parameters()).device
        # Initialize entire buffer as pure noise
        generated = torch.randn(
            batch_size, self.buf_len, *self.spatial_shape, self.input_dim, device=device
        )
        generated = [generated[i] for i in range(batch_size)]
        self.generated = self.preprocess(generated)

        # Initialize streaming state for cross module
        self.text_module.init_stream(self.batch_size)
        self._init_stream_condition(
            batch_size,
            self.buf_len,
            device,
            history_position=history_position,
            history_latent=history_latent,
        )

        # Optional: inject GT history with triangular re-noising matching schedule at t=1
        if history_latent is not None:
            h = history_latent[0].shape[0]
            chunk_size = self.time_scheduler.chunk_size
            # Triangular ts: ts[0]=1 (clean), ts[chunk_size-1]=1/chunk_size (most noisy),
            # ts[i>=chunk_size]=0 (full noise — beyond effective transition zone)
            ts_history = torch.clamp(
                1 - torch.arange(h, device=device, dtype=torch.float32) / chunk_size,
                min=0.0, max=1.0,
            )
            alpha_h, _, beta_h, _, _, _, _ = self.time_scheduler.get_noise_levels(
                device, [h], [ts_history],
            )
            alpha_b = alpha_h[0][None, :, None, None]
            beta_b = beta_h[0][None, :, None, None]
            for i in range(batch_size):
                z = (history_latent[i].to(device) - self.mean) / self.std
                z = self.preprocess([z])[0]
                noise = torch.randn_like(z)
                z_noised = z * alpha_b + noise * beta_b
                self.generated[i][:, :h, ...] = z_noised
            # Conditioned subclasses may need to restore channels that must
            # remain clean after the base model's triangular history re-noising.
            self._apply_stream_state_condition()
            # t starts from 1 → wavefront just past the history transition zone
            self.current_step = self.time_scheduler.steps

    def _rollback(self):
        """Shift buffer by seq_len when conditions overflow the window."""
        for i in range(self.batch_size):
            self.generated[i][:, : self.seq_len, ...] = self.generated[i][
                :, self.seq_len :, ...
            ].clone()
            self.generated[i][:, self.seq_len :, ...] = torch.randn_like(
                self.generated[i][:, self.seq_len :, ...]
            )
        self.current_step -= self.time_scheduler.get_step_rollback(self.seq_len)
        self.condition_frames -= self.seq_len
        self.current_commit -= self.seq_len
        self.text_module.trim_stream(self.seq_len)
        self._rollback_stream_condition(self.seq_len)

    @torch.no_grad()
    def stream_generate_step(self, x):
        """
        Streaming generation step. Each call provides 1 frame of conditions.
        The scheduler determines committable frames from accumulated conditions.

        Returns:
            dict with "generated": list of one (N, C) tensor, or [] if nothing to commit.
        """
        x = self._extract_inputs(x)
        device = next(self.parameters()).device
        self.generated = [g.to(device) for g in self.generated]

        # 1. Update conditions (1 frame per call)
        self._update_stream_condition(x, device)
        self.text_module.update_stream(x, device, self.param_dtype)
        self.condition_frames += 1

        # 2. Rollback if conditions overflow the window
        if self.condition_frames > self.buf_len:
            self._rollback()
        self._commit_stream_condition(self.condition_frames - 1)

        # 3. Determine how many frames can be committed
        committable_length, committable_steps = self.time_scheduler.get_committable(
            self.condition_frames
        )
        while self.current_step < committable_steps:
            s = self.time_scheduler.prepare(
                self.generated, device, [self.buf_len] * self.batch_size,
                current_step=self.current_step,
            )
            time_schedules = s["time_schedules"]
            time_schedules_derivative = s["time_schedules_derivative"]
            alpha = s["alpha"]
            dalpha = s["dalpha"]
            beta = s["beta"]
            dbeta = s["dbeta"]
            sigma = s["sigma"]
            dlog_alpha = s["dlog_alpha"]
            dlog_beta = s["dlog_beta"]
            is_ = s["input_start"]
            ie_ = s["input_end"]
            os_ = s["output_start"]
            oe_ = s["output_end"]
            xt = s["xt"]
            xt = self._condition_stream_input(xt, is_, ie_)

            # time_schedules already sliced to input window by prepare()
            time_schedules_input = [
                time_schedules[0] * self.time_embedding_scale
            ] * self.batch_size

            # CFG: batch text + null in one forward pass
            text_context, text_meta = self.text_module.get_stream_context(
                is_[0], ie_[0], device, self.param_dtype,
            )
            text_mask = text_meta["cross_attn_mask"]
            text_k_rope = text_meta.get("text_k_rope_ids")
            # Q-rope override (per-sample List[Tuple|None]): real-text stream
            # entries are None and pass through _shift unchanged.
            text_q_rope = text_meta.get("text_q_rope_ids")

            window_len = ie_[0] - is_[0]
            null_context, null_meta = self.text_module.get_null_context(
                text_context, device, self.param_dtype,
                valid_len=[window_len] * self.batch_size,
            )
            null_mask = null_meta["cross_attn_mask"]
            null_text_k_rope = null_meta.get("text_k_rope_ids")
            null_text_q_rope = null_meta.get("text_q_rope_ids")

            ts_patch = self._expand_schedule_to_patch(time_schedules_input)
            ctx_patch = self._expand_context_to_patch(text_context + null_context)
            cross_mask_patch = self._expand_mask_to_patch(text_mask + null_mask)
            # K-rope ids: real is buffer-absolute → shift to window-relative; null
            # comes from get_null_context already window-relative (zeros).
            text_k_rope = self._shift_rope_ids(
                text_k_rope, [is_[0]] * self.batch_size,
            )
            text_q_rope = self._shift_rope_ids(
                text_q_rope, [is_[0]] * self.batch_size,
            )
            if text_k_rope is not None and null_text_k_rope is not None:
                text_k_rope_patch = text_k_rope + null_text_k_rope
            else:
                text_k_rope_patch = None
            # Q-override: real samples carry None (cross uses self_attn rope),
            # null samples carry zeros tuple (phase=0 vs K=0).
            if text_q_rope is not None or null_text_q_rope is not None:
                text_q_rope_patch = (
                    (text_q_rope if text_q_rope is not None
                     else [None] * self.batch_size)
                    + (null_text_q_rope if null_text_q_rope is not None
                       else [None] * self.batch_size)
                )
            else:
                text_q_rope_patch = None

            if self.attn_type == "partial":
                masks = self._build_partial_mask(is_, ie_, os_, oe_, device)
                attn_mask = masks + masks   # CFG doubled batch (text + null), same mask
            else:
                attn_mask = None
            stream_text_seq_len = max(
                int(c.size(0)) for c in (text_context + null_context)
            )
            pred_all = self.model(
                xt + xt,
                ts_patch + ts_patch,
                ctx_patch,
                self._patch_seq_len(self.seq_len),
                attn_mask=attn_mask,
                text_k_rope_ids=text_k_rope_patch,
                text_q_rope_ids=text_q_rope_patch,
                cross_attn_mask=cross_mask_patch,
                text_seq_len=stream_text_seq_len,
                y=None,
            )
            pred_text = pred_all[: self.batch_size]
            pred_null = pred_all[self.batch_size :]
            predicted_result = [
                self.cfg_config["text_scale"] * pt + self.cfg_config["null_scale"] * pn
                for pt, pn in zip(pred_text, pred_null)
            ]

            # All noise coefficients already sliced to output window by prepare()
            os_idx, oe_idx = os_[0], oe_[0]
            pred_os_idx = os_idx - is_[0]
            pred_oe_idx = oe_idx - is_[0]
            dt = time_schedules_derivative[0][None, :, None, None]
            alpha_i = alpha[0][None, :, None, None]
            dalpha_i = dalpha[0][None, :, None, None]
            beta_i = beta[0][None, :, None, None]
            dbeta_i = dbeta[0][None, :, None, None]
            sigma_i = sigma[0][None, :, None, None]
            dlog_alpha_i = dlog_alpha[0][None, :, None, None]
            dlog_beta_i = dlog_beta[0][None, :, None, None]
            for i in range(self.batch_size):
                predicted_result_i = predicted_result[i][:, pred_os_idx:pred_oe_idx, ...]
                generated_i = self.generated[i][:, os_idx:oe_idx, ...]
                if self.prediction_type == "vel":
                    vel = predicted_result_i
                elif self.prediction_type == "x0":
                    vel = (
                        predicted_result_i * (-dlog_beta_i * alpha_i + dalpha_i)
                        + generated_i * dlog_beta_i
                    )
                elif self.prediction_type == "eps":
                    vel = (
                        predicted_result_i * (-dlog_alpha_i * beta_i + dbeta_i)
                        + generated_i * dlog_alpha_i
                    )
                st = (vel - generated_i * dlog_alpha_i) / (
                    (beta_i * dlog_alpha_i - dbeta_i) * beta_i
                )
                self.generated[i][:, os_idx:oe_idx, ...] += (
                    vel * dt
                    + st * 0.5 * sigma_i**2 * dt
                    + sigma_i * torch.sqrt(dt) * torch.randn_like(generated_i)
                )
            self._apply_stream_state_condition()
            self.current_step += 1

        # 5. Extract newly committed frames
        if self.current_commit < committable_length:
            commit_start = self.current_commit
            output = [
                self.generated[i][:, commit_start:committable_length, ...]
                for i in range(self.batch_size)
            ]
            output = self.postprocess(output)
            output = [o * self.std + self.mean for o in output]
            output = self._finalize_stream_output(
                output, commit_start, committable_length,
            )
            self.current_commit = committable_length
            result = {"generated": output}
            result.update(
                self._stream_output_metadata(commit_start, committable_length)
            )
            return result
        else:
            empty = [
                torch.zeros(self.input_dim, 0, *self.spatial_shape, device=device)
                for _ in range(self.batch_size)
            ]
            empty = self.postprocess(empty)
            empty = self._finalize_stream_output(
                empty, self.current_commit, self.current_commit,
            )
            result = {"generated": empty}
            result.update(
                self._stream_output_metadata(
                    self.current_commit, self.current_commit,
                )
            )
            return result

# Position-specific root packing and model contracts.
from .vae_wan_position import (
    POSITION_ROOT_DIMS,
    WAN_ROOT_GROUP_SIZE,
    pack_root_condition,
    resolve_position_representation,
)

# Channels that are identically zero in the ground truth have std ~0.  Dividing
# by that std turns float noise into an O(1) training target, so below this
# floor the channel is centred but not scaled.  The floor sits in the measured
# gap observed in the supported motion statistics.
STD_NORMALIZATION_FLOOR = 1e-3


def guard_std(std, floor=STD_NORMALIZATION_FLOOR):
    """Replace near-zero standard deviations with 1 (i.e. do not divide)."""
    std = torch.as_tensor(std).float()
    return torch.where(std < floor, torch.ones_like(std), std)


def _to_batched_tensor(value, device, dtype=torch.float32):
    if isinstance(value, (list, tuple)):
        tensors = [torch.as_tensor(v, device=device, dtype=dtype) for v in value]
        if not tensors:
            raise ValueError("position condition cannot be empty")
        if tensors[0].ndim == 0:
            if any(tensor.ndim != 0 for tensor in tensors):
                raise ValueError("position list mixes scalar and non-scalar values")
            return torch.stack(tensors, dim=0)
        if tensors[0].ndim == 1:
            return torch.stack(tensors, dim=0)
        lengths = {int(t.size(0)) for t in tensors}
        if len(lengths) != 1:
            raise ValueError(
                "variable-length position lists must be padded/collated before use"
            )
        return torch.stack(tensors, dim=0)
    return torch.as_tensor(value, device=device, dtype=dtype)


class _PositionStreamBufferMixin:
    """Fixed-size, rollback-safe raw condition buffer for stream generation."""

    stream_condition_dim = 0

    def _assert_stream_schedule(self):
        """Subclass hook: validated again after init_generated rebuilds it."""
        return None

    def init_generated(self, seq_len, *args, **kwargs):
        # The base rebuilds time_scheduler from a caller-supplied
        # schedule_config, so schedule invariants must be re-checked.
        result = super().init_generated(seq_len, *args, **kwargs)
        self._assert_stream_schedule()
        return result

    def _init_stream_condition(
        self,
        batch_size,
        buf_len,
        device,
        history_position=None,
        history_latent=None,
    ):
        self._stream_position = torch.zeros(
            batch_size,
            buf_len,
            self.stream_condition_dim,
            device=device,
            dtype=torch.float32,
        )
        self._stream_position_valid = torch.zeros(
            batch_size, buf_len, device=device, dtype=torch.bool
        )
        self._pending_stream_position = None

        if history_position is not None and history_latent is None:
            raise ValueError("history_position requires matching history_latent")
        history_length = None
        if history_latent is not None:
            if len(history_latent) != batch_size:
                raise ValueError(
                    f"history latent batch={len(history_latent)}, expected {batch_size}"
                )
            history_lengths = [int(torch.as_tensor(v).size(0)) for v in history_latent]
            if len(set(history_lengths)) != 1:
                raise ValueError(
                    f"all history latent lengths must match, got {history_lengths}"
                )
            history_length = history_lengths[0]
            if history_length > buf_len:
                raise ValueError(
                    f"history latent length={history_length} exceeds buffer={buf_len}"
                )

        if history_position is None and history_latent is not None:
            history_position = self._position_from_history(history_latent)
        if history_position is not None:
            history = self._coerce_stream_history(history_position, device)
            if history.size(0) != batch_size:
                raise ValueError(
                    f"history position batch={history.size(0)}, expected {batch_size}"
                )
            if history.size(1) > buf_len:
                raise ValueError(
                    f"history position length={history.size(1)} exceeds buffer={buf_len}"
                )
            if history_length is not None and history.size(1) != history_length:
                raise ValueError(
                    f"history position length={history.size(1)} does not match "
                    f"history latent length={history_length}"
                )
            self._stream_position[:, : history.size(1)] = history
            self._stream_position_valid[:, : history.size(1)] = True

    def _position_from_history(self, history_latent):
        return None

    def _coerce_stream_history(self, position, device):
        value = _to_batched_tensor(position, device)
        if value.ndim == 2:
            value = value.unsqueeze(0)
        if value.ndim != 3 or value.size(-1) != self.stream_condition_dim:
            raise ValueError(
                f"history position must be (B,T,{self.stream_condition_dim}), "
                f"got {tuple(value.shape)}"
            )
        if not torch.isfinite(value).all():
            raise ValueError("history position contains NaN or Inf")
        return value.float()

    def _update_stream_condition(self, inputs, device):
        # Validation happens before text/counters are mutated in the base class.
        value = self._parse_stream_position(inputs, device)
        if value.shape != (self.batch_size, self.stream_condition_dim):
            raise ValueError(
                f"stream position must resolve to "
                f"({self.batch_size},{self.stream_condition_dim}), "
                f"got {tuple(value.shape)}"
            )
        if not torch.isfinite(value).all():
            raise ValueError("stream position contains NaN or Inf")
        self._pending_stream_position = value.float()

    def _commit_stream_condition(self, index):
        if self._pending_stream_position is None:
            raise RuntimeError("stream position validation/commit transaction is missing")
        if not 0 <= index < self.buf_len:
            raise IndexError(
                f"stream position index {index} outside buffer [0,{self.buf_len})"
            )
        self._stream_position[:, index] = self._pending_stream_position
        self._stream_position_valid[:, index] = True
        self._pending_stream_position = None

    def _rollback_stream_condition(self, trim_len):
        keep = self.buf_len - trim_len
        self._stream_position[:, :keep] = self._stream_position[
            :, trim_len:
        ].clone()
        self._stream_position_valid[:, :keep] = self._stream_position_valid[
            :, trim_len:
        ].clone()
        self._stream_position[:, keep:] = 0
        self._stream_position_valid[:, keep:] = False

    def _stream_position_slice(self, start, end):
        valid = self._stream_position_valid[:, start:end]
        if valid.numel() and not bool(valid.all()):
            missing = (~valid).nonzero(as_tuple=False)[0].tolist()
            raise RuntimeError(
                f"stream root condition missing inside active window "
                f"[{start},{end}); first missing batch/local-index={missing}"
            )
        return self._stream_position[:, start:end]


class _RootConditionedDiffusion(
    _PositionStreamBufferMixin, DiffForcingWanModel
):
    """Diffusion whose root channels are pure conditioning.

    The root is never noised, never predicted and never scored: the diffused
    state holds only the non-root channels, the root is concatenated in front
    of the denoiser input at every training window and every denoise step, and
    the emitted output is ``[supplied root, network state]``.  Root output is
    therefore bit-exact with the supplied root.

    Subclasses define how a batch's raw root maps onto one condition vector per
    state unit (``condition_dim``): the direct model uses one root vector per
    frame, while the latent model packs four root vectors per latent token.
    """

    condition_dim = 0

    def _split_statistics(self, mean_path, std_path, expected_width):
        """Split one statistics pair into (condition, state) halves.

        Both halves come from the same file, so the condition is normalized
        with per-dimension statistics of the exact values it carries.
        """
        if (mean_path is None) != (std_path is None):
            raise ValueError("mean_path and std_path must be provided together")
        mean, std = load_statistics(mean_path, std_path, expected_width)
        mean, std = mean.reshape(-1), std.reshape(-1)
        if mean.numel() != expected_width or std.numel() != expected_width:
            raise ValueError(
                f"statistics must have {expected_width} values, got "
                f"{mean.numel()}/{std.numel()}"
            )
        std = guard_std(std)
        return (
            mean[: self.condition_dim].clone(),
            std[: self.condition_dim].clone(),
            mean[self.condition_dim :].clone(),
            std[self.condition_dim :].clone(),
        )

    def _register_statistics(self, mean_path, std_path, expected_width):
        cond_mean, cond_std, state_mean, state_std = self._split_statistics(
            mean_path, std_path, expected_width
        )
        self.mean = state_mean
        self.std = state_std
        self.register_buffer("position_mean", cond_mean)
        self.register_buffer("position_std", cond_std)

    def _assert_stream_schedule(self):
        """Reject schedules that commit frames without ever denoising them.

        ``get_committable`` yields zero committable steps when
        ``steps < chunk_size``, so the loop body never runs while frames are
        still emitted.
        """
        steps = int(self.time_scheduler.steps)
        chunk_size = int(self.time_scheduler.chunk_size)
        if steps < chunk_size or steps % chunk_size != 0:
            raise ValueError(
                "streaming requires steps to be a positive multiple of "
                f"chunk_size, got steps={steps}, chunk_size={chunk_size}"
            )

    # ------------------------------------------------------------------
    # Condition extraction (subclass hooks)
    # ------------------------------------------------------------------
    def _split_feature(self, feature):
        """Return (condition_part, state_part) of a full-width feature."""
        raise NotImplementedError

    def _pack_position_batch(self, position, valid_len, seq_len, device):
        """Return (B, seq_len, condition_dim) raw conditions."""
        raise NotImplementedError

    def _extract_inputs(self, x):
        inputs = super()._extract_inputs(x)
        # Position-only and streaming callers commonly use the literal key
        # without adding it to input_keys.  Keep custom mappings authoritative,
        # then fall back to the public ``position`` key.
        if "position" not in inputs:
            position_key = self.input_keys.get("position", "position")
            if position_key in x:
                inputs["position"] = x[position_key]
            elif "position" in x:
                inputs["position"] = x["position"]
        feature = inputs.get("feature")
        if feature is not None:
            width = feature.size(-1)
            if width == self.condition_dim + self.input_dim:
                condition, state = self._split_feature(feature)
                inputs["feature"] = state
                inputs.setdefault("position", condition)
            elif width != self.input_dim:
                raise ValueError(
                    f"feature must have {self.condition_dim + self.input_dim} "
                    f"(root+state) or {self.input_dim} (state) channels, got {width}"
                )
        return inputs

    def _get_raw_position(self, inputs):
        position = inputs.get("position")
        if position is None:
            raise KeyError(
                "root conditioning is required: supply input_keys.position or a "
                "full-width feature whose leading channels carry the root"
            )
        return position

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _prepare_training_condition(
        self, inputs, normalized_feature, valid_len, device
    ):
        packed = self._pack_position_batch(
            self._get_raw_position(inputs),
            valid_len,
            normalized_feature.size(1),
            device,
        )
        normalized = (packed - self.position_mean) / self.position_std
        return [
            normalized[i].transpose(0, 1)[:, :, None, None]
            for i in range(normalized.size(0))
        ]

    def _condition_training_input(
        self, noisy, clean, condition, batch_index, start, end
    ):
        root = condition[batch_index][:, start:end]
        return torch.cat([root.to(noisy), noisy], dim=0)

    # ------------------------------------------------------------------
    # Offline generation
    # ------------------------------------------------------------------
    def _prepare_generation_condition(self, inputs, valid_len, seq_len, device):
        packed = self._pack_position_batch(
            self._get_raw_position(inputs), valid_len, seq_len, device
        )
        normalized = (packed - self.position_mean) / self.position_std
        return {
            "raw": packed,
            "normalized": [
                normalized[i].transpose(0, 1)[:, :, None, None]
                for i in range(normalized.size(0))
            ],
        }

    def _condition_generation_input(self, noisy, condition, starts, ends):
        return [
            torch.cat(
                [
                    condition["normalized"][i][:, starts[i] : ends[i]].to(noisy[i]),
                    noisy[i],
                ],
                dim=0,
            )
            for i in range(len(noisy))
        ]

    def _finalize_generation_output(self, generated, condition, valid_len):
        return [
            torch.cat(
                [condition["raw"][i, : valid_len[i]].to(generated[i]), generated[i]],
                dim=-1,
            )
            for i in range(len(generated))
        ]

    def _generation_output_metadata(self, condition, valid_len):
        return {
            "position": [
                condition["raw"][i, : valid_len[i]].clone()
                for i in range(len(valid_len))
            ]
        }

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------
    def _condition_stream_input(self, noisy, starts, ends):
        result = []
        for i, item in enumerate(noisy):
            raw = self._stream_position_slice(starts[i], ends[i])[i]
            norm = (raw - self.position_mean) / self.position_std
            root = norm.transpose(0, 1)[:, :, None, None].to(item)
            result.append(torch.cat([root, item], dim=0))
        return result

    def _finalize_stream_output(self, generated, start, end):
        raw = self._stream_position_slice(start, end)
        return [
            torch.cat([raw[i].to(generated[i]), generated[i]], dim=-1)
            for i in range(len(generated))
        ]

    def _stream_output_metadata(self, start, end):
        # Clone: the slice is a view of the rollback-mutated ring buffer.
        raw = self._stream_position_slice(start, end)
        return {"position": [raw[i].clone() for i in range(self.batch_size)]}


class DiffForcingPositionWanModel(_RootConditionedDiffusion):
    """Direct full-feature diffusion conditioned on clean root motion.

    ``input_dim`` keeps its config meaning (the complete feature width); the
    diffused state contains every non-root channel.  The denoiser reads the
    clean root plus noisy pose state and predicts only the pose state, so the
    loss never covers root motion and emitted root channels are exactly the
    supplied values.
    """

    def __init__(
        self,
        *args,
        input_dim=138,
        representation=None,
        root_dim=None,
        mean_path=None,
        std_path=None,
        root_mean_path=None,
        root_std_path=None,
        **kwargs,
    ):
        self.feature_dim = int(input_dim)
        self.representation, self.root_dim = resolve_position_representation(
            representation,
            feature_dim=self.feature_dim,
            root_dim=root_dim,
        )
        self.condition_dim = self.root_dim
        if not 0 < self.root_dim < self.feature_dim:
            raise ValueError(
                f"root_dim must be in (0,{self.feature_dim}), got {self.root_dim}"
            )
        kwargs.pop("model_input_dim", None)
        super().__init__(
            *args,
            input_dim=self.feature_dim - self.root_dim,
            model_input_dim=self.feature_dim,
            mean_path=None,
            std_path=None,
            **kwargs,
        )
        self._assert_stream_schedule()
        self.stream_condition_dim = self.root_dim
        self._register_statistics(mean_path, std_path, self.feature_dim)
        if root_mean_path is not None or root_std_path is not None:
            # Optional per-dimension root statistics can override the leading
            # values from the full-feature statistics.
            if root_mean_path is None or root_std_path is None:
                raise ValueError(
                    "root_mean_path and root_std_path must be provided together"
                )
            root_mean, root_std = load_statistics(
                root_mean_path, root_std_path, self.root_dim
            )
            root_mean, root_std = root_mean.reshape(-1), root_std.reshape(-1)
            if root_mean.numel() < self.root_dim or root_std.numel() < self.root_dim:
                raise ValueError(
                    f"root statistics must have at least {self.root_dim} values"
                )
            self.position_mean = root_mean[: self.root_dim].clone()
            self.position_std = guard_std(root_std[: self.root_dim]).clone()

    def _split_feature(self, feature):
        return feature[..., : self.root_dim], feature[..., self.root_dim :]

    def _pack_position_batch(self, position, valid_len, seq_len, device):
        position = _to_batched_tensor(position, device)
        if position.ndim == 2:
            position = position.unsqueeze(0)
        if position.ndim != 3 or position.size(-1) < self.root_dim:
            raise ValueError(
                f"direct root must be (B,T,>={self.root_dim}), got "
                f"{tuple(position.shape)}"
            )
        if position.size(0) != len(valid_len):
            raise ValueError(
                f"root batch={position.size(0)}, expected {len(valid_len)}"
            )
        raw = position[..., : self.root_dim].float()
        padded = []
        for i, length in enumerate(valid_len):
            if raw.size(1) < length:
                raise ValueError(
                    f"sample {i} has {raw.size(1)} root frames but needs {length}"
                )
            item = raw[i, :length]
            if seq_len > length:
                if length == 0:
                    raise ValueError("cannot pad an empty valid root sequence")
                item = torch.cat([item, item[-1:].expand(seq_len - length, -1)], dim=0)
            padded.append(item)
        packed = torch.stack(padded)
        if not torch.isfinite(packed).all():
            raise ValueError("root condition contains NaN or Inf")
        return packed

    def _position_from_history(self, history_latent):
        tensors = [torch.as_tensor(v) for v in history_latent]
        if not tensors:
            return None
        width = int(tensors[0].size(-1))
        if width == self.feature_dim:
            return torch.stack([v[..., : self.root_dim] for v in tensors])
        return None

    def init_generated(
        self,
        seq_len,
        batch_size=1,
        schedule_config={},
        history_latent=None,
        history_position=None,
    ):
        if history_latent is not None:
            # A full-width history carries its own root, which must be pulled
            # out here: the diffused state has no root channels for the base
            # class to recover it from.
            split_state = []
            inferred_position = []
            for item in history_latent:
                item = torch.as_tensor(item)
                width = int(item.size(-1))
                if width == self.feature_dim:
                    inferred_position.append(item[..., : self.root_dim])
                    split_state.append(item[..., self.root_dim :])
                elif width == self.input_dim:
                    split_state.append(item)
                else:
                    raise ValueError(
                        f"history must be {self.feature_dim}D (root+state) or "
                        f"{self.input_dim}D (state), got {width}D"
                    )
            if inferred_position:
                if len(inferred_position) != len(split_state):
                    raise ValueError("cannot mix full-width and state-only history")
                if history_position is not None:
                    raise ValueError(
                        "history_position must be omitted for a full-width history"
                    )
                history_position = inferred_position
            elif history_position is None:
                raise ValueError("a state-only history requires history_position")
            history_latent = split_state
        return super().init_generated(
            seq_len,
            batch_size=batch_size,
            schedule_config=schedule_config,
            history_latent=history_latent,
            history_position=history_position,
        )

    def _parse_stream_position(self, inputs, device):
        position = _to_batched_tensor(self._get_raw_position(inputs), device)
        if position.ndim == 3 and position.size(1) == 1:
            position = position[:, 0]
        if position.ndim == 1:
            position = position.unsqueeze(0)
        if position.ndim != 2 or position.size(-1) < self.root_dim:
            raise ValueError(
                f"each direct stream step needs (B,{self.root_dim}) root, "
                f"got {tuple(position.shape)}"
            )
        return position[..., : self.root_dim]


class LatentDiffForcingPositionWanModel(_RootConditionedDiffusion):
    """Latent diffusion conditioned on the WAN-causal packed root.

    ``input_dim`` keeps its config meaning (the VAE latent width); the diffused
    state is exactly that latent.  The denoiser reads ``position_dim + latent``
    channels and predicts only the latent, so the emitted token is
    ``[supplied packed root, network latent]`` and its root is bit-exact.

    The packed root width is inferred, in order, from ``representation``, an
    explicit ``root_dim``/``position_dim``, or the token statistics width.  The
    last path keeps existing configs working when they do not forward the
    top-level representation into model parameters.
    """

    def __init__(
        self,
        *args,
        input_dim=16,
        representation=None,
        position_dim=None,
        root_dim=None,
        root_group_size=WAN_ROOT_GROUP_SIZE,
        mean_path=None,
        std_path=None,
        **kwargs,
    ):
        self.root_group_size = int(root_group_size)
        if self.root_group_size != WAN_ROOT_GROUP_SIZE:
            raise ValueError(
                "the WAN position ABI groups exactly "
                f"{WAN_ROOT_GROUP_SIZE} root frames per latent, got "
                f"root_group_size={self.root_group_size}"
            )
        self.latent_dim = int(input_dim)
        if self.latent_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.latent_dim}")

        resolved_root_dim = None if root_dim is None else int(root_dim)
        self.representation = None
        if representation is not None:
            self.representation, resolved_root_dim = (
                resolve_position_representation(
                    representation,
                    root_dim=resolved_root_dim,
                )
            )

        resolved_position_dim = (
            None if position_dim is None else int(position_dim)
        )
        if resolved_position_dim is not None:
            if resolved_position_dim <= 0:
                raise ValueError(
                    f"position_dim must be positive, got {resolved_position_dim}"
                )
            if resolved_position_dim % self.root_group_size != 0:
                raise ValueError(
                    "position_dim must be divisible by root_group_size "
                    f"({self.root_group_size}), got {resolved_position_dim}"
                )
            position_root_dim = resolved_position_dim // self.root_group_size
            if (
                resolved_root_dim is not None
                and resolved_root_dim != position_root_dim
            ):
                raise ValueError(
                    "position_dim and root_dim disagree: "
                    f"{resolved_position_dim} != {resolved_root_dim} * "
                    f"{self.root_group_size}"
                )
            resolved_root_dim = position_root_dim
        elif resolved_root_dim is not None:
            resolved_position_dim = resolved_root_dim * self.root_group_size
        elif (
            mean_path is not None and std_path is not None
            and Path(mean_path).is_file() and Path(std_path).is_file()
        ):
            mean_width = int(np.asarray(np.load(mean_path)).size)
            std_width = int(np.asarray(np.load(std_path)).size)
            if mean_width != std_width:
                raise ValueError(
                    "token statistics widths must match, got "
                    f"{mean_width}/{std_width}"
                )
            resolved_position_dim = mean_width - self.latent_dim
            if (
                resolved_position_dim <= 0
                or resolved_position_dim % self.root_group_size != 0
            ):
                raise ValueError(
                    f"cannot infer packed root width from {mean_width}D token "
                    f"statistics, latent_dim={self.latent_dim}, and "
                    f"root_group_size={self.root_group_size}"
                )
            resolved_root_dim = resolved_position_dim // self.root_group_size
            if resolved_root_dim not in set(POSITION_ROOT_DIMS.values()):
                raise ValueError(
                    f"cannot infer a supported root width from {mean_width}D "
                    f"token statistics: inferred root_dim={resolved_root_dim}, "
                    f"supported root widths are "
                    f"{sorted(set(POSITION_ROOT_DIMS.values()))}; pass "
                    "root_dim/position_dim explicitly for a custom schema"
                )
        else:
            # Historical root3/root12 default for configs without token stats.
            _, resolved_root_dim = resolve_position_representation()
            resolved_position_dim = resolved_root_dim * self.root_group_size

        # If a representation supplied the root width, also validate any width
        # inferred from position_dim or token statistics.
        self.representation, resolved_root_dim = resolve_position_representation(
            self.representation,
            root_dim=resolved_root_dim,
        )
        self.root_dim = resolved_root_dim
        self.position_dim = resolved_position_dim
        self.condition_dim = self.position_dim
        if self.position_dim != self.root_dim * self.root_group_size:
            raise ValueError(
                "position_dim must equal root_dim * root_group_size "
                f"({self.root_dim}*{self.root_group_size})"
            )
        kwargs.pop("model_input_dim", None)
        super().__init__(
            *args,
            input_dim=self.latent_dim,
            model_input_dim=self.position_dim + self.latent_dim,
            mean_path=None,
            std_path=None,
            **kwargs,
        )
        self._assert_stream_schedule()
        self.stream_condition_dim = self.position_dim
        self._register_statistics(
            mean_path, std_path, self.position_dim + self.latent_dim
        )

    def _split_feature(self, feature):
        return feature[..., : self.position_dim], feature[..., self.position_dim :]

    def _pack_position_batch(self, position, valid_len, seq_len, device):
        position = _to_batched_tensor(position, device)
        if position.ndim == 2:
            position = position.unsqueeze(0)
        if position.ndim != 3:
            raise ValueError(
                f"latent root must be (B,L,{self.position_dim}) or (B,T,>="
                f"{self.root_dim}), got {tuple(position.shape)}"
            )
        if position.size(0) != len(valid_len):
            raise ValueError(
                f"root batch={position.size(0)}, expected {len(valid_len)}"
            )
        packed_items = []
        for i, length in enumerate(valid_len):
            item = position[i : i + 1]
            if item.size(-1) == self.position_dim:
                if item.size(1) < length:
                    raise ValueError(
                        f"sample {i} has {item.size(1)} packed roots but needs {length}"
                    )
                packed = item[:, :length]
            else:
                required_frames = (
                    0 if length == 0 else 1 + self.root_group_size * (length - 1)
                )
                if item.size(-1) < self.root_dim or item.size(1) < required_frames:
                    raise ValueError(
                        f"sample {i} needs {required_frames} raw root frames for "
                        f"{length} latents, got {item.size(1)}"
                    )
                # Trim to this sample's own frames: in a collated batch the
                # tensor is padded to the batch maximum, and pack_root_condition
                # clamps its taps to the frame count it is given, so an untrimmed
                # item would let a short sample's tail taps land on pad zeros.
                item = item[:, :required_frames]
                packed = pack_root_condition(
                    item[..., : self.root_dim],
                    latent_length=length,
                    root_dim=self.root_dim,
                    group_size=self.root_group_size,
                    first_chunk=True,
                )
            packed = packed[0]
            if seq_len > length:
                if length == 0:
                    raise ValueError("cannot pad an empty valid root sequence")
                packed = torch.cat(
                    [packed, packed[-1:].expand(seq_len - length, -1)], dim=0
                )
            packed_items.append(packed)
        packed = torch.stack(packed_items).float()
        if not torch.isfinite(packed).all():
            raise ValueError("latent root condition contains NaN or Inf")
        return packed

    def _position_from_history(self, history_latent):
        tensors = [torch.as_tensor(v) for v in history_latent]
        if not tensors:
            return None
        width = int(tensors[0].size(-1))
        if width == self.position_dim + self.latent_dim:
            return torch.stack([v[..., : self.position_dim] for v in tensors])
        return None

    def init_generated(
        self,
        seq_len,
        batch_size=1,
        schedule_config={},
        history_latent=None,
        history_position=None,
    ):
        if history_latent is not None:
            token_dim = self.position_dim + self.latent_dim
            split_state = []
            inferred_position = []
            for item in history_latent:
                item = torch.as_tensor(item)
                width = int(item.size(-1))
                if width == token_dim:
                    inferred_position.append(item[..., : self.position_dim])
                    split_state.append(item[..., self.position_dim :])
                elif width == self.latent_dim:
                    split_state.append(item)
                else:
                    raise ValueError(
                        f"latent history must be {token_dim}D (root+latent) or "
                        f"{self.latent_dim}D (latent), got {width}D"
                    )
            if inferred_position:
                if len(inferred_position) != len(split_state):
                    raise ValueError("cannot mix token-width and latent-only history")
                if history_position is not None:
                    raise ValueError(
                        "history_position must be omitted for a token-width history"
                    )
                history_position = inferred_position
            elif history_position is None:
                raise ValueError("a latent-only history requires history_position")
            history_latent = split_state
        return super().init_generated(
            seq_len,
            batch_size=batch_size,
            schedule_config=schedule_config,
            history_latent=history_latent,
            history_position=history_position,
        )

    def _parse_stream_position(self, inputs, device):
        position = _to_batched_tensor(self._get_raw_position(inputs), device)
        if position.ndim == 3 and position.shape[-2:] == (
            self.root_group_size,
            self.root_dim,
        ):
            position = position.flatten(-2)
        elif position.ndim == 3 and position.size(1) == 1:
            position = position[:, 0]
        if position.ndim == 1:
            position = position.unsqueeze(0)
        if position.ndim != 2 or position.size(-1) != self.position_dim:
            raise ValueError(
                f"each latent stream step needs (B,{self.position_dim}) or "
                f"(B,{self.root_group_size},{self.root_dim}) root, got "
                f"{tuple(position.shape)}"
            )
        return position


# Concise aliases for configs and downstream imports.
DiffForcingWanPositionModel = DiffForcingPositionWanModel
LatentDiffForcingWanPositionModel = LatentDiffForcingPositionWanModel
