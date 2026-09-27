# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified to drop flash_attn dependency. partial-mask paths can't use flash_attn,
# so all callers route through torch SDPA.
import torch

__all__ = ["attention"]


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    dtype=torch.bfloat16,
    attn_mask=None,
):
    """SDPA-based attention matching the previous flash_attn varlen wrapper.

    Args:
        q: (B, Lq, Nq, D)
        k: (B, Lk, Nk, D)
        v: (B, Lk, Nk, D)
        q_lens: not supported (raises NotImplementedError if provided).
        k_lens: optional (B,) per-sample valid K length.
        attn_mask: optional (B, Lq, Lk) bool mask, AND-ed with length / causal /
                   window mask. Broadcasts over heads.
        causal, window_size: bottom-right aligned to the qL_b × kL_b sub-matrix
                             (matches flash_attn varlen semantics).

    Output: (B, Lq, Nq, D), restored to q.dtype. Fully-masked rows softmax to
    NaN under SDPA; we replace them with zeros to match flash_attn's behavior.
    """
    if q_lens is not None:
        raise NotImplementedError("variable q_lens not supported")

    half_dtypes = (torch.float16, torch.bfloat16)
    out_dtype = q.dtype
    B, Lq, _, _ = q.shape
    Lk = k.shape[1]

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    q = half(q)
    k = half(k)
    v = half(v)
    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    # SDPA expects (B, N, L, D)
    q_t = q.transpose(1, 2).contiguous()
    k_t = k.transpose(1, 2).contiguous()
    v_t = v.transpose(1, 2).contiguous()

    needs_mask = (
        k_lens is not None
        or window_size != (-1, -1)
        or attn_mask is not None
    )
    if needs_mask:
        device = q.device
        i = torch.arange(Lq, device=device).view(1, Lq, 1)
        j = torch.arange(Lk, device=device).view(1, 1, Lk)

        # q_lens guarded by NotImplementedError above, so all Q positions valid.
        q_valid = torch.ones(B, Lq, 1, dtype=torch.bool, device=device)
        if k_lens is not None:
            if k_lens.device.type == "cpu":
                # Shape metadata stays on the host. Scalar comparisons avoid a
                # blocking host-to-device copy inside CUDA Graph capture.
                k_valid = torch.cat([j < length for length in k_lens.tolist()], dim=0)
            else:
                kL = k_lens.to(device=device, dtype=torch.long).view(B, 1, 1)
                k_valid = j < kL
        else:
            k_valid = torch.ones(B, 1, Lk, dtype=torch.bool, device=device)
        mask = q_valid & k_valid  # (B, Lq, Lk)

        # Standard top-left aligned causal/window (matches PyTorch SDPA
        # convention): Q[i] attends to K[j] when j <= i, with the optional
        # window_size further restricting (j - i) into [-left, right].
        if causal:
            mask = mask & (j <= i)

        left, right = window_size
        if left != -1:
            mask = mask & ((j - i) >= -left)
        if right != -1:
            mask = mask & ((j - i) <= right)

        if attn_mask is not None:
            user_mask = attn_mask.to(dtype=torch.bool, device=q.device)
            assert user_mask.dim() == 3, (
                f"attn_mask must be (B, Lq, Lk) bool, got shape {tuple(user_mask.shape)}"
            )
            mask = mask & user_mask

        attn_mask_b = mask.unsqueeze(1)  # (B, 1, Lq, Lk), broadcast over heads
        is_causal = False  # already encoded in mask
    else:
        attn_mask_b = None
        is_causal = causal

    out = torch.nn.functional.scaled_dot_product_attention(
        q_t,
        k_t,
        v_t,
        attn_mask=attn_mask_b,
        is_causal=is_causal,
        dropout_p=dropout_p,
        scale=softmax_scale,
    )
    # Fully-masked rows softmax to NaN under SDPA; flash_attn emits zeros.
    out = torch.nan_to_num(out, nan=0.0)
    out = out.transpose(1, 2).contiguous()  # (B, Lq, Nq, D)
    return out.type(out_dtype)
