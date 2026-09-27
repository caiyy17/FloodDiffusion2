"""Shared T2M evaluation metric (263-D evaluator pipeline).

The evaluator network is trained on HumanML3D-263, so every representation is
funneled to 263 first. Per-rep wiring is a thin subclass in
metrics/<Rep>/t2m.py that plugs its own single-sequence funnel:

    class T2MMetrics(T2MMetricsBase):
        to_humanml3d_single = staticmethod(to_humanml3d)   # (L, D) -> (L', 263)

This file is representation-agnostic metric infrastructure.
"""

from typing import List

import numpy as np
import torch
from lightning.pytorch.utilities import rank_zero_warn
from torch import Tensor
from torchmetrics import Metric

from utils.initialize import instantiate
from .utils import (
    calculate_activation_statistics_np,
    calculate_diversity_np,
    calculate_frechet_distance_np,
    calculate_top_k,
    euclidean_distance_matrix,
)


class T2MMetricsBase(Metric):
    # per-rep hook: (L, D) float array -> (L', 263) float array
    to_humanml3d_single = None

    def __init__(
        self,
        cfg,
        top_k=3,
        R_size=32,
        diversity_times=300,
        dist_sync_on_step=False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step, sync_on_compute=False)

        self.cfg = cfg
        self.evaluate_text = self.cfg.evaluate_text

        self.top_k = top_k
        self.R_size = R_size
        # cfg.diversity_times (yaml) overrides the kwarg default when present —
        # `instantiate` passes cfg as a single positional arg so explicit fields
        # in yaml otherwise wouldn't reach this attribute.
        self.diversity_times = self.cfg.get("diversity_times", diversity_times)

        # Metric names
        self.metrics = []
        if self.evaluate_text:
            self.Matching_metrics = ["Matching_score", "gt_Matching_score"]
            for k in range(1, top_k + 1):
                self.Matching_metrics.append(f"R_precision_top_{str(k)}")
            for k in range(1, top_k + 1):
                self.Matching_metrics.append(f"gt_R_precision_top_{str(k)}")
            self.metrics.extend(self.Matching_metrics)

        self.metrics.extend(["FID", "Diversity", "gt_Diversity"])

        # Cached batches
        self.failed_batches = 0
        self.add_state("text_embeddings", default=[], dist_reduce_fx=None)
        self.add_state("recmotion_embeddings", default=[], dist_reduce_fx=None)
        self.add_state("gtmotion_embeddings", default=[], dist_reduce_fx=None)

        # Load 263D evaluator
        self._get_t2m_evaluator(self.cfg)

    def _get_t2m_evaluator(self, cfg):
        """Load T2M evaluator (same 263D evaluator as HumanML3D263)."""
        t2m_checkpoint = {}
        if self.evaluate_text:
            self.w_vectorizer = instantiate(
                cfg.wordvectorizer.target, cfg=None, **cfg.wordvectorizer.params,
            )
            self.t2m_textencoder = instantiate(
                cfg.textencoder.target, cfg=None, **cfg.textencoder.params,
            )
            t2m_checkpoint["text_encoder"] = torch.load(
                cfg.textencoder.ckpt, map_location="cpu",
            )
            self.t2m_textencoder.load_state_dict(t2m_checkpoint["text_encoder"])
            self.t2m_textencoder.eval()
            for p in self.t2m_textencoder.parameters():
                p.requires_grad = False

        # 263D normalization statistics for the evaluator
        self.metric_mean_np = np.load(cfg.metric_mean_path)
        self.metric_std_np = np.load(cfg.metric_std_path)
        self.register_buffer(
            "metric_mean", torch.from_numpy(self.metric_mean_np).float()
        )
        self.register_buffer("metric_std", torch.from_numpy(self.metric_std_np).float())

        self.t2m_moveencoder = instantiate(
            cfg.moveencoder.target, cfg=None, **cfg.moveencoder.params,
        )
        self.t2m_motionencoder = instantiate(
            cfg.motionencoder.target, cfg=None, **cfg.motionencoder.params,
        )

        t2m_checkpoint["movement_encoder"] = torch.load(
            cfg.moveencoder.ckpt, map_location="cpu",
        )
        t2m_checkpoint["motion_encoder"] = torch.load(
            cfg.motionencoder.ckpt, map_location="cpu",
        )
        self.t2m_moveencoder.load_state_dict(t2m_checkpoint["movement_encoder"])
        self.t2m_motionencoder.load_state_dict(t2m_checkpoint["motion_encoder"])
        self.t2m_moveencoder.eval()
        self.t2m_motionencoder.eval()
        for p in self.t2m_moveencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_motionencoder.parameters():
            p.requires_grad = False

    def _batch_to_humanml3d(self, feats, lengths):
        """Convert a batch of rep features to HumanML3D 263D via the rep funnel.

        Args:
            feats: (B, T, D) tensor on any device.
            lengths: list of int, actual lengths per sample.

        Returns:
            feats_263: (B, T', 263) tensor on the same device.
            new_lengths: list of int (funnel-dependent, e.g. L-1 or L).
        """
        if self.to_humanml3d_single is None:
            raise NotImplementedError("subclass must set to_humanml3d_single")
        device = feats.device
        B = feats.shape[0]
        feats_np = feats.float().cpu().numpy()

        converted = []
        new_lengths = []
        for i in range(B):
            data_263 = np.asarray(self.to_humanml3d_single(feats_np[i, :lengths[i]]))
            converted.append(data_263)
            new_lengths.append(data_263.shape[0])

        # Pad and stack
        max_len = max(new_lengths)
        padded = np.zeros((B, max_len, 263), dtype=np.float32)
        for i, arr in enumerate(converted):
            padded[i, :arr.shape[0]] = arr

        return torch.from_numpy(padded).to(device), new_lengths

    def reset(self):
        super().reset()
        self.failed_batches = 0

    @torch.no_grad()
    def compute(self, sanity_flag=False):
        metrics = {}
        metrics_device = self.device

        # Gather embeddings from all GPUs FIRST — every rank must join the
        # collectives even with zero local successes, or ranks deadlock.
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            gathered_recmotion = [None] * world_size
            gathered_gtmotion = [None] * world_size
            torch.distributed.all_gather_object(gathered_recmotion, self.recmotion_embeddings)
            torch.distributed.all_gather_object(gathered_gtmotion, self.gtmotion_embeddings)
            self.recmotion_embeddings = [e for rank in gathered_recmotion for e in rank]
            self.gtmotion_embeddings = [e for rank in gathered_gtmotion for e in rank]
            if self.evaluate_text:
                gathered_text = [None] * world_size
                torch.distributed.all_gather_object(gathered_text, self.text_embeddings)
                self.text_embeddings = [e for rank in gathered_text for e in rank]

        if not self.recmotion_embeddings:
            # No successful batch: report NaN (0.0 would make a broken
            # evaluator look perfect), except in the sanity pass.
            fill = 0.0 if sanity_flag else float("nan")
            if not sanity_flag:
                rank_zero_warn(
                    f"T2MMetrics.compute: no successful batches "
                    f"({self.failed_batches} failed) — reporting NaN"
                )
            if self.evaluate_text:
                for metric in self.Matching_metrics:
                    metrics[metric] = torch.tensor(fill, device=metrics_device)
            metrics["FID"] = torch.tensor(fill, device=metrics_device)
            metrics["Diversity"] = torch.tensor(fill, device=metrics_device)
            metrics["gt_Diversity"] = torch.tensor(fill, device=metrics_device)
            self.reset()
            return metrics

        count_seq = len(self.recmotion_embeddings)
        shuffle_idx = torch.randperm(count_seq)
        all_genmotions = torch.cat(self.recmotion_embeddings, axis=0)[shuffle_idx, :]
        all_gtmotions = torch.cat(self.gtmotion_embeddings, axis=0)[shuffle_idx, :]

        # Text-related metrics (Matching_score, R_precision_top_k).
        # On failure: set all matching metrics to NaN and continue with FID/Diversity.
        if self.evaluate_text:
            try:
                all_texts = torch.cat(self.text_embeddings, axis=0)[shuffle_idx, :]
                if count_seq >= self.R_size:
                    matching_score_sum = 0.0
                    top_k_mat = torch.zeros((self.top_k,))
                    for i in range(count_seq // self.R_size):
                        s, e = i * self.R_size, (i + 1) * self.R_size
                        dist_mat = euclidean_distance_matrix(all_texts[s:e], all_genmotions[s:e]).nan_to_num()
                        if dist_mat.dtype == torch.bfloat16:
                            dist_mat = dist_mat.float()
                        matching_score_sum += dist_mat.trace()
                        argsmax = torch.argsort(dist_mat, dim=1)
                        top_k_mat += calculate_top_k(argsmax, top_k=self.top_k).sum(axis=0)

                    R_count = count_seq // self.R_size * self.R_size
                    metrics["Matching_score"] = (matching_score_sum / R_count).detach().clone().to(metrics_device)
                    for k in range(self.top_k):
                        metrics[f"R_precision_top_{str(k + 1)}"] = (top_k_mat[k] / R_count).detach().clone().to(metrics_device)

                    gt_matching_score_sum = 0.0
                    top_k_mat = torch.zeros((self.top_k,))
                    for i in range(count_seq // self.R_size):
                        s, e = i * self.R_size, (i + 1) * self.R_size
                        dist_mat = euclidean_distance_matrix(all_texts[s:e], all_gtmotions[s:e]).nan_to_num()
                        if dist_mat.dtype == torch.bfloat16:
                            dist_mat = dist_mat.float()
                        gt_matching_score_sum += dist_mat.trace()
                        argsmax = torch.argsort(dist_mat, dim=1)
                        top_k_mat += calculate_top_k(argsmax, top_k=self.top_k).sum(axis=0)
                    metrics["gt_Matching_score"] = (gt_matching_score_sum / R_count).detach().clone().to(metrics_device)
                    for k in range(self.top_k):
                        metrics[f"gt_R_precision_top_{str(k + 1)}"] = (top_k_mat[k] / R_count).detach().clone().to(metrics_device)
                else:
                    for metric in self.Matching_metrics:
                        metrics[metric] = torch.tensor(0.0, device=metrics_device)
            except Exception as exc:
                rank_zero_warn(f"T2MMetrics matching block failed; setting NaN: {exc!r}")
                for metric in self.Matching_metrics:
                    metrics[metric] = torch.tensor(float("nan"), device=metrics_device)

        # FID
        all_genmotions = all_genmotions.float().numpy()
        all_gtmotions = all_gtmotions.float().numpy()
        mu, cov = calculate_activation_statistics_np(all_genmotions)
        gt_mu, gt_cov = calculate_activation_statistics_np(all_gtmotions)
        try:
            fid_value = calculate_frechet_distance_np(gt_mu, gt_cov, mu, cov)
        except ValueError:
            fid_value = float('inf')
        metrics["FID"] = torch.tensor(fid_value, dtype=torch.float32, device=metrics_device)

        # Diversity. Wrap each direction independently — gen embeddings may have
        # nan/inf from a divergent generate() while gt are still clean.
        # `calculate_diversity_np` uses scipy.linalg.norm which raises on nan/inf.
        if count_seq >= self.diversity_times:
            try:
                metrics["Diversity"] = torch.tensor(
                    calculate_diversity_np(all_genmotions, self.diversity_times),
                    dtype=torch.float32, device=metrics_device,
                )
            except Exception as exc:
                rank_zero_warn(f"Diversity (gen) failed; setting NaN: {exc!r}")
                metrics["Diversity"] = torch.tensor(float("nan"), device=metrics_device)
            try:
                metrics["gt_Diversity"] = torch.tensor(
                    calculate_diversity_np(all_gtmotions, self.diversity_times),
                    dtype=torch.float32, device=metrics_device,
                )
            except Exception as exc:
                rank_zero_warn(f"gt_Diversity failed; setting NaN: {exc!r}")
                metrics["gt_Diversity"] = torch.tensor(float("nan"), device=metrics_device)
        else:
            metrics["Diversity"] = torch.tensor(0.0, device=metrics_device)
            metrics["gt_Diversity"] = torch.tensor(0.0, device=metrics_device)

        for key in metrics:
            if isinstance(metrics[key], torch.Tensor) and metrics[key].device != metrics_device:
                metrics[key] = metrics[key].to(metrics_device)

        self.reset()
        return metrics

    @torch.no_grad()
    def update(
        self,
        feats_ref: Tensor,
        feats_rst: Tensor,
        lengths_ref: List[int],
        lengths_rst: List[int],
        text_tokens: List[List[str]] = None,
    ):
        # All-or-nothing per batch: if any step fails (138->263 conversion, motion
        # encoder forward, text encoder forward), drop the whole batch instead of
        # leaving rec/gt/text lists out of sync. Subsequent batches keep going.
        try:
            # Convert rep features -> HumanML3D 263D
            feats_ref_263, lengths_ref_263 = self._batch_to_humanml3d(feats_ref, lengths_ref)
            feats_rst_263, lengths_rst_263 = self._batch_to_humanml3d(feats_rst, lengths_rst)

            # Encode ground truth motions (263D -> embeddings)
            align_idx = np.argsort(lengths_ref_263)[::-1].copy()
            feats_ref_sorted = feats_ref_263[align_idx]
            lengths_ref_sorted = np.array(lengths_ref_263)[align_idx]
            gtmotion_embeddings = self._get_motion_embeddings(feats_ref_sorted, lengths_ref_sorted)
            gt_cache = [0] * len(lengths_ref_263)
            for i in range(len(lengths_ref_263)):
                gt_cache[align_idx[i]] = gtmotion_embeddings[i : i + 1].cpu()

            # Encode generated motions
            align_idx = np.argsort(lengths_rst_263)[::-1].copy()
            feats_rst_sorted = feats_rst_263[align_idx]
            lengths_rst_sorted = np.array(lengths_rst_263)[align_idx]
            recmotion_embeddings = self._get_motion_embeddings(feats_rst_sorted, lengths_rst_sorted)
            rec_cache = [0] * len(lengths_rst_263)
            for i in range(len(lengths_rst_263)):
                rec_cache[align_idx[i]] = recmotion_embeddings[i : i + 1].cpu()

            # Text embeddings (only collected if everything above succeeded)
            text_cache = []
            if self.evaluate_text:
                for tokens in text_tokens:
                    max_text_len = self.cfg.wordvectorizer.max_text_len
                    if len(tokens) < max_text_len:
                        tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
                        sent_len = len(tokens)
                        tokens = tokens + ["unk/OTHER"] * (max_text_len + 2 - sent_len)
                    else:
                        tokens = tokens[:max_text_len]
                        tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
                        sent_len = len(tokens)

                    pos_one_hots = []
                    word_embeddings = []
                    for token in tokens:
                        word_emb, pos_oh = self.w_vectorizer[token]
                        pos_one_hots.append(pos_oh[None, :])
                        word_embeddings.append(word_emb[None, :])
                    pos_one_hots = np.concatenate(pos_one_hots, axis=0)
                    word_embeddings = np.concatenate(word_embeddings, axis=0)

                    pos_one_hots = torch.from_numpy(pos_one_hots.astype(np.float32)).to(self.device)
                    word_embeddings = torch.from_numpy(word_embeddings.astype(np.float32)).to(self.device)
                    text_lengths = torch.tensor(sent_len).to(self.device)

                    self.t2m_textencoder = self.t2m_textencoder.to(self.device)
                    text_emb = self.t2m_textencoder(
                        word_embeddings[None, ...],
                        pos_one_hots[None, ...],
                        text_lengths[None, ...],
                    )
                    text_embeddings = torch.flatten(text_emb, start_dim=1).detach()
                    text_cache.append(text_embeddings.cpu())
        except (TypeError, AttributeError, KeyError, NameError, ImportError,
                NotImplementedError, AssertionError):
            raise    # programming/configuration errors must surface, not be skipped
        except Exception as exc:
            self.failed_batches += 1
            rank_zero_warn(
                f"T2MMetrics.update skipped one batch ({self.failed_batches} failed so far): {exc!r}"
            )
            return

        # Commit batch state atomically.
        self.gtmotion_embeddings.extend(gt_cache)
        self.recmotion_embeddings.extend(rec_cache)
        if self.evaluate_text:
            self.text_embeddings.extend(text_cache)

    def _get_motion_embeddings(self, feats: Tensor, lengths):
        """Normalize 263D features and encode with the 263D evaluator."""
        device = feats.device
        feats = (feats - self.metric_mean) / self.metric_std

        self.t2m_moveencoder = self.t2m_moveencoder.to(device)
        self.t2m_motionencoder = self.t2m_motionencoder.to(device)

        m_lens = torch.tensor(lengths)
        m_lens = torch.div(m_lens, 4, rounding_mode="floor")
        mov = self.t2m_moveencoder(feats[..., :-4]).detach()
        emb = self.t2m_motionencoder(mov, m_lens)

        return torch.flatten(emb, start_dim=1).detach()
