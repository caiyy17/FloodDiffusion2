"""Chunk-based diffusion model -- NEW version.

Inherits the multi-window pack training paradigm from
diffusion_forcing_wan_new.DiffForcingWanModel. Replaces the schedule with
a chunk-based one (each chunk has a fixed window; history segment schedule
= 1.0 clean, target segment schedule = t_frac).

Config: history_len=m, chunk_size=n, steps=T
- Global time t in [0, num_chunks), num_chunks = 1 + ceil((N - (m+n)) / n)
- Schedule: before window 1.0 / history 1.0 (clean) / target t_frac / after 0.0
- Training and inference both keep history clean; only target frames are denoised
- Under multi-window training, W_k sees all preceding clean frames in
  clean[0:s_k] (a feature, not strictly limited to history_len)
- For attn_type in {full, causal}, __init__ (in the parent) forces
  train_n_windows=1
"""

import math

import numpy as np
import torch

from .diffusion_forcing_wan import DiffForcingWanModel

EPSILON = 0.05


class ChunkDiffusionScheduler:
    """Chunk schedule with the NEW scheduler contract (exposes get_t_max +
    slice_input; prepare is the inference-side one-shot helper)."""

    def __init__(self, config):
        self.steps = config["steps"]
        self.chunk_size = config["chunk_size"]  # n
        self.history_len = config.get("history_len", 0)  # m
        self.window_size = self.history_len + self.chunk_size  # m+n
        self.noise_type = config.get("noise_type", "linear")
        self.sigma_type = config.get("sigma_type", "zero")
        self.random_epsilon = config.get("random_epsilon", 0.0)
        self.content_len = config.get("content_len", None)
        # True (default): num_chunks includes a trailing partial chunk.
        # False: full chunks only; cover the tail with extra_len padding
        # (typically chunk_size-1) at generate time.
        self.t_max_tail = bool(config.get("t_max_tail", True))

        if self.noise_type in ("exponential", "exponential_rev"):
            self.exp_max = config.get("exp_max", 5.0)
        elif self.noise_type == "diffusion":
            self.T = config.get("T", 1000)
            self.beta_start = config.get("beta_start", 0.0001)
            self.beta_end = config.get("beta_end", 0.02)

        if self.sigma_type == "memoryless":
            self.sigma_scale = config.get("sigma_scale", 1.0)

    # ---------------- chunk geometry ---------------- #

    def _num_chunks(self, seq_len):
        if self.t_max_tail:
            if seq_len <= self.window_size:
                return 1
            return 1 + math.ceil((seq_len - self.window_size) / self.chunk_size)
        # full chunks only
        n = 0
        if seq_len >= self.window_size:
            n = 1 + (seq_len - self.window_size) // self.chunk_size
        assert n >= 1, (
            f"seq_len {seq_len} < window_size {self.window_size} yields no "
            f"full chunk with t_max_tail=False; pad with extra_len or set "
            f"t_max_tail=true"
        )
        return n

    def _window_range(self, seq_len, chunk_idx):
        """Return (input_start, input_end, output_start, output_end) for a chunk."""
        assert seq_len > self.history_len, (
            f"sample length {seq_len} <= history_len {self.history_len}: "
            f"chunk 0 window would be inverted (nothing to denoise); "
            f"filter short samples in the dataloader"
        )
        if chunk_idx == 0:
            os_ = self.history_len           # the first m frames are GT history
            oe_ = min(self.window_size, seq_len)
            is_ = 0
        else:
            os_ = self.window_size + (chunk_idx - 1) * self.chunk_size
            oe_ = min(os_ + self.chunk_size, seq_len)
            is_ = os_ - self.history_len
        if self.content_len is not None:
            is_ = max(is_, oe_ - self.content_len)
        return is_, oe_, os_, oe_

    # ---------------- NEW scheduler contract ---------------- #

    def get_t_max(self, valid_len):
        """Upper bound of t = number of chunks."""
        return self._num_chunks(valid_len)

    def get_total_steps(self, seq_len):
        return self._num_chunks(seq_len) * self.steps

    def get_time_steps(self, device, valid_len, current_step):
        """current_step (int) -> t scalar per sample. Inference-only."""
        t = current_step * (1.0 / self.steps)
        return [torch.tensor(t, device=device) for _ in range(len(valid_len))]

    def get_time_schedules(self, device, valid_len, time_steps, training=False):
        time_schedules = []
        time_schedules_derivative = []
        for i in range(len(valid_len)):
            t = time_steps[i].item()
            chunk_idx = min(int(t), self._num_chunks(valid_len[i]) - 1)
            t_frac = t - chunk_idx
            _is, _ie, os_, oe_ = self._window_range(valid_len[i], chunk_idx)

            ts = torch.zeros(valid_len[i], device=device)
            ts[:os_] = 1.0                   # before window + history → clean
            ts[os_:oe_] = t_frac             # target → fractional time
            tsd = torch.full((valid_len[i],), 1.0 / self.steps, device=device)
            if training:
                ts = torch.clamp(
                    ts + torch.randn_like(ts) * self.random_epsilon,
                    min=0.0, max=1.0,
                )
            time_schedules.append(ts)
            time_schedules_derivative.append(tsd)
        return time_schedules, time_schedules_derivative

    def get_windows(self, valid_len, time_steps, training=False):
        # `training` is part of the scheduler contract; chunk windows are
        # integer geometry with no eps, so it is unused.
        input_start, input_end, output_start, output_end = [], [], [], []
        for i in range(len(time_steps)):
            t = time_steps[i].item()
            chunk_idx = min(int(t), self._num_chunks(valid_len[i]) - 1)
            is_, ie_, os_, oe_ = self._window_range(valid_len[i], chunk_idx)
            input_start.append(is_)
            input_end.append(ie_)
            output_start.append(os_)
            output_end.append(oe_)
        return input_start, input_end, output_start, output_end

    def get_noise_levels(self, device, valid_len, time_schedules):
        alpha, dalpha, dlog_alpha = [], [], []
        beta, dbeta, dlog_beta = [], [], []
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
                k = self.exp_max
                alpha_i = torch.exp(-k * (1 - t))
                dalpha_i = k * alpha_i
                dlog_alpha_i = k * torch.ones_like(alpha_i)
                beta_i = 1 - alpha_i
                dbeta_i = -dalpha_i
                dlog_beta_i = dbeta_i / torch.clamp(beta_i, min=EPSILON)
            elif self.noise_type == "exponential_rev":
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
                if self.noise_type in ("linear", "exponential", "exponential_rev"):
                    sigma_i = self.sigma_scale * torch.sqrt(
                        torch.clamp(2 * dlog_alpha_i * beta_i, min=0.0)
                    )
                elif self.noise_type == "diffusion":
                    sigma_i = self.sigma_scale * torch.sqrt(
                        torch.clamp(2 * dlog_alpha_i, min=0.0)
                    )
                else:
                    sigma_i = self.sigma_scale * torch.sqrt(
                        torch.clamp(2 * beta_i * (dlog_alpha_i * beta_i - dbeta_i), min=0.0)
                    )
            sigma.append(sigma_i)
        return alpha, dalpha, beta, dbeta, sigma, dlog_alpha, dlog_beta

    def slice_input(self, x, input_start, input_end):
        """Inference-side: slice each sample's current state to its input window."""
        return [x[i][:, input_start[i]:input_end[i], ...] for i in range(len(x))]

    def prepare(self, x, device, valid_len, current_step):
        """Inference-side one-shot helper."""
        time_steps = self.get_time_steps(device, valid_len, current_step)
        time_schedules, time_schedules_derivative = self.get_time_schedules(
            device, valid_len, time_steps,
        )
        alpha, dalpha, beta, dbeta, sigma, dlog_alpha, dlog_beta = \
            self.get_noise_levels(device, valid_len, time_schedules)
        input_start, input_end, output_start, output_end = \
            self.get_windows(valid_len, time_steps)
        xt = self.slice_input(x, input_start, input_end)

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
            "alpha": alpha, "dalpha": dalpha, "beta": beta, "dbeta": dbeta,
            "sigma": sigma, "dlog_alpha": dlog_alpha, "dlog_beta": dlog_beta,
            "xt": xt,
        }

    # ---------------- Streaming support ---------------- #

    def get_committable(self, total_frames):
        if total_frames < self.window_size:
            return 0, 0
        committed = self.window_size
        committable_steps = self.steps
        remaining = total_frames - self.window_size
        extra_chunks = remaining // self.chunk_size
        committed += extra_chunks * self.chunk_size
        committable_steps += extra_chunks * self.steps
        return committed, committable_steps

    def get_step_rollback(self, seq_len):
        return (seq_len // self.chunk_size) * self.steps


class ChunkDiffWanModel(DiffForcingWanModel):
    """Chunk-based diffusion model. History is always kept clean; only the
    target frames get denoised.

    The first chunk uses GT history (history_len frames) as conditioning;
    subsequent chunks use previously generated frames as history. Supports
    NEW's multi-window training (attn_type='partial') or single-window
    training (n_windows is forced to 1 by __init__ when attn_type is
    'full' or 'causal').
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.time_scheduler = ChunkDiffusionScheduler(self.schedule_config)

    def generate(self, x):
        """Chunk inference: inject GT history at the start, then denoise
        chunk by chunk."""
        x = self._extract_inputs(x)
        feature_length = x["feature_length"]
        batch_size = len(feature_length)
        # extra_len pads the rollout so all real frames fall inside full
        # chunks when t_max_tail=False (typically chunk_size-1).
        extra_len = self.schedule_config.get("extra_len", 0)
        seq_len = max(feature_length).item() + extra_len
        if (not self.time_scheduler.t_max_tail
                and extra_len < self.time_scheduler.chunk_size - 1):
            import warnings
            warnings.warn(
                f"chunk t_max_tail=False with extra_len={extra_len} < "
                f"chunk_size-1={self.time_scheduler.chunk_size - 1}: trailing "
                f"frames may not be covered by any full chunk",
                stacklevel=2,
            )
        device = next(self.parameters()).device
        valid_len = [min(fl.item(), seq_len) for fl in feature_length]
        generated_len = [seq_len] * batch_size
        history_len = self.time_scheduler.history_len
        assert min(valid_len) > history_len, (
            f"feature_length {valid_len} contains samples <= history_len "
            f"{history_len}: their entire output would be injected GT history, "
            f"nothing generated; filter short samples"
        )

        # Initialize the full-length buffer with pure noise
        generated = torch.randn(
            batch_size, seq_len, *self.spatial_shape, self.input_dim, device=device
        )
        generated = [generated[i] for i in range(batch_size)]
        generated = self.preprocess(generated)

        # Inject GT history into the first history_len frames of the buffer
        if "feature" in x:
            gt_feature = x["feature"]
            gt_feature = (gt_feature - self.mean) / self.std
            gt_list = [gt_feature[i, :valid_len[i], ...] for i in range(batch_size)]
            gt_list = self.preprocess(gt_list)
            for i in range(batch_size):
                h = min(history_len, gt_list[i].shape[1])
                generated[i][:, :h, ...] = gt_list[i][:, :h, ...]

        # T5 context (real)
        text_context, metadata = self.text_module.get_context(
            x, generated_len, device, self.param_dtype, training=False,
        )
        full_text = metadata["full_text"]
        cross_attn_mask_full = metadata["cross_attn_mask"]
        text_k_rope_ids_full = metadata.get("text_k_rope_ids")

        total_steps = self.time_scheduler.get_total_steps(seq_len)
        for step in range(total_steps):
            s = self.time_scheduler.prepare(
                generated, device, generated_len, current_step=step,
            )
            time_schedules = s["time_schedules"]
            time_schedules_derivative = s["time_schedules_derivative"]
            alpha = s["alpha"]; dalpha = s["dalpha"]
            beta = s["beta"]; dbeta = s["dbeta"]
            sigma = s["sigma"]
            dlog_alpha = s["dlog_alpha"]; dlog_beta = s["dlog_beta"]
            is_idx = s["input_start"]; ie_idx = s["input_end"]
            os_idx_all = s["output_start"]; oe_idx_all = s["output_end"]
            xt = s["xt"]

            time_schedules_input = [
                time_schedules[i] * self.time_embedding_scale for i in range(batch_size)
            ]

            window_cross_attn_mask = [
                cross_attn_mask_full[i][is_idx[i]:ie_idx[i], :] for i in range(batch_size)
            ]
            window_text_k_rope = self._shift_rope_ids(text_k_rope_ids_full, is_idx)

            ts_patch = self._expand_schedule_to_patch(time_schedules_input)
            patch_seq_len = self._patch_seq_len(seq_len)

            # CFG: text + null doubled batch
            window_lens = [ie_idx[i] - is_idx[i] for i in range(batch_size)]
            window_null_context, window_null_meta = self.text_module.get_null_context(
                text_context, device, self.param_dtype, valid_len=window_lens,
            )
            window_null_mask = window_null_meta["cross_attn_mask"]
            window_null_text_k_rope = window_null_meta.get("text_k_rope_ids")
            window_null_text_q_rope = window_null_meta.get("text_q_rope_ids")

            combined_ctx = self._expand_context_to_patch(text_context + window_null_context)
            combined_cross_mask = self._expand_mask_to_patch(
                window_cross_attn_mask + window_null_mask
            )
            if window_text_k_rope is not None and window_null_text_k_rope is not None:
                combined_text_k_rope = window_text_k_rope + window_null_text_k_rope
            else:
                combined_text_k_rope = None
            if window_null_text_q_rope is not None:
                combined_text_q_rope = [None] * batch_size + list(window_null_text_q_rope)
            else:
                combined_text_q_rope = None

            attn_mask = (
                self._build_partial_mask(is_idx, ie_idx, os_idx_all, oe_idx_all, device)
                if self.attn_type == "partial" else None
            )
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
                cross_attn_mask=combined_cross_mask,
                text_seq_len=combined_text_seq_len,
                y=None,
            )
            pred_text = pred_all[:batch_size]
            pred_null = pred_all[batch_size:]
            predicted_result = [
                self.cfg_config["text_scale"] * pt + self.cfg_config["null_scale"] * pn
                for pt, pn in zip(pred_text, pred_null)
            ]

            # SDE update on target frames only
            for i in range(batch_size):
                os_i, oe_i = os_idx_all[i], oe_idx_all[i]
                pred_os = os_i - is_idx[i]
                pred_oe = oe_i - is_idx[i]
                predicted_result_i = predicted_result[i][:, pred_os:pred_oe, ...]
                generated_i = generated[i][:, os_i:oe_i, ...]
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
                generated[i][:, os_i:oe_i, ...] += (
                    vel * dt
                    + st * 0.5 * sigma_i ** 2 * dt
                    + sigma_i * torch.sqrt(dt) * torch.randn_like(generated_i)
                )

        generated = self.postprocess(generated)
        y_hat_out = []
        for i in range(batch_size):
            single_generated = generated[i][:valid_len[i], :] * self.std + self.mean
            y_hat_out.append(single_generated)
        return {"generated": y_hat_out, "text": full_text}

    def init_generated(self, seq_len, batch_size=1, schedule_config={}, history_latent=None):
        """Stream init: inject GT history into the first history_len frames
        of the buffer."""
        super().init_generated(seq_len, batch_size, schedule_config)
        self.time_scheduler = ChunkDiffusionScheduler(self.schedule_config)
        if history_latent is not None:
            m = self.time_scheduler.history_len
            device = next(self.parameters()).device
            for i in range(batch_size):
                z = (history_latent[i].to(device) - self.mean) / self.std
                z = self.preprocess([z])[0]
                h = min(m, z.shape[1])
                self.generated[i][:, :h, ...] = z[:, :h, ...]
