"""Standard diffusion model (non-forcing) -- NEW version.

Inherits diffusion_forcing_wan_new.DiffForcingWanModel. All frames share
the same noise level (uniform t) with no windowing.

Multi-window training (n>1) is meaningless for standard diffusion -- every
W_k covers the full sequence, so N>1 just multiplies compute N times.
__init__ forces train_n_windows=1.
"""

import numpy as np
import torch

from .diffusion_forcing_wan import DiffForcingWanModel

EPSILON = 0.05


class DiffusionScheduler:
    """Standard diffusion scheduler — uniform noise level across all frames。
    NEW scheduler contract: exposes get_t_max + slice_input; prepare is the
    inference-side one-shot helper."""

    def __init__(self, config):
        self.steps = config["steps"]
        self.noise_type = config.get("noise_type", "linear")
        self.sigma_type = config.get("sigma_type", "zero")
        self.random_epsilon = config.get("random_epsilon", 0.0)

        if self.noise_type in ("exponential", "exponential_rev"):
            self.exp_max = config.get("exp_max", 5.0)
        elif self.noise_type == "diffusion":
            self.T = config.get("T", 1000)
            self.beta_start = config.get("beta_start", 0.0001)
            self.beta_end = config.get("beta_end", 0.02)

        if self.sigma_type == "memoryless":
            self.sigma_scale = config.get("sigma_scale", 1.0)

    def get_t_max(self, valid_len):
        return 1.0    # diffusion t ∈ [0, 1]

    def get_total_steps(self, seq_len):
        return self.steps

    def get_time_steps(self, device, valid_len, current_step):
        t = current_step * (1.0 / self.steps)
        return [torch.tensor(t, device=device) for _ in range(len(valid_len))]

    def get_time_schedules(self, device, valid_len, time_steps, training=False):
        time_schedules = []
        time_schedules_derivative = []
        for i in range(len(valid_len)):
            t = time_steps[i].item()
            ts = torch.full((valid_len[i],), t, device=device)
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
        """Standard diffusion: window always covers the full sequence.
        `training` is part of the scheduler contract; no eps here, so unused."""
        input_start = [0] * len(time_steps)
        input_end = list(valid_len)
        output_start = [0] * len(time_steps)
        output_end = list(valid_len)
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
        return [x[i][:, input_start[i]:input_end[i], ...] for i in range(len(x))]

    def prepare(self, x, device, valid_len, current_step):
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
        # All windows = full sequence for diffusion, slicing is no-op
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


class DiffusionWanModel(DiffForcingWanModel):
    """Standard diffusion (uniform t across all frames, no windowing)。

    Multi-window training (n>1) is meaningless for standard diffusion (every
    W_k covers the full sequence). __init__ forces train_n_windows=1.
    """

    def __init__(self, **kwargs):
        sc = kwargs.setdefault("schedule_config", {})
        sc.setdefault("chunk_size", 1)
        n = int(sc.get("train_n_windows", 1))
        if n != 1:
            import warnings
            warnings.warn(
                f"diffusion has no windowing concept; forcing train_n_windows "
                f"to 1 (N>1 would inflate packed length to N*T_i and make "
                f"self-attn quadratic). Original value {n} -> 1.",
                stacklevel=2,
            )
        sc["train_n_windows"] = 1
        super().__init__(**kwargs)
        self.time_scheduler = DiffusionScheduler(self.schedule_config)
