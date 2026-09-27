import os

import numpy as np
import torch
from lightning import seed_everything
from torch_ema import ExponentialMovingAverage

from utils.initialize import compare_statedict_and_parameters, instantiate, load_config
from visualization.visualize import render_single_video

# Set tokenizers parallelism to false to avoid warnings in multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class IdentityVAE:
    """VAE stand-in for pure-df models (config has no test_vae): the model
    generates features directly, so decoding is the identity and there is no
    temporal down/upsampling (1 committed token == 1 frame)."""

    def __init__(self, input_dim):
        self.input_dim = input_dim

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    def clear_cache(self):
        pass

    def stream_decode(self, x, first_chunk=True):
        return x

    def decode(self, x):
        return x


def load_model_from_config():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    cfg = load_config()
    seed_everything(cfg.seed)

    if "test_vae" not in cfg.config:
        # pure df: the model generates features directly, no VAE
        print("No test_vae in config — using IdentityVAE (pure df)")
        vae = IdentityVAE(cfg.model.params["input_dim"])
        return vae, _load_diffusion(cfg), cfg

    vae = instantiate(
        target=cfg.test_vae.target,
        cfg=None,
        hfstyle=False,
        **cfg.test_vae.params,
    )
    vae_ckpt = torch.load(cfg.test_vae_ckpt, map_location="cpu", weights_only=False)
    if "ema_state" in vae_ckpt:
        vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
        vae_ema = ExponentialMovingAverage(
            vae.parameters(), decay=cfg.test_vae.ema_decay
        )
        vae_ema.load_state_dict(vae_ckpt["ema_state"])
        vae_ema.copy_to(vae.parameters())
        print(f"Loaded VAE model from {cfg.test_vae_ckpt} with EMA")
    else:
        vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
        print(f"Loaded VAE model from {cfg.test_vae_ckpt} w/o EMA")

    compare_statedict_and_parameters(
        state_dict=vae.state_dict(),
        named_parameters=vae.named_parameters(),
        named_buffers=vae.named_buffers(),
    )
    vae.to(device)
    vae.eval()

    return vae, _load_diffusion(cfg), cfg


def _load_diffusion(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = instantiate(
        target=cfg.model.target, cfg=None, hfstyle=False, **cfg.model.params
    )
    checkpoint = torch.load(cfg.test_ckpt, map_location="cpu", weights_only=False)
    if "ema_state" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        ema = ExponentialMovingAverage(model.parameters(), decay=cfg.model.ema_decay)
        ema.load_state_dict(checkpoint["ema_state"])
        ema.copy_to(model.parameters())
        print(f"Loaded model from {cfg.test_ckpt} with EMA")
    else:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        print(f"Loaded model from {cfg.test_ckpt} w/o EMA")

    compare_statedict_and_parameters(
        state_dict=model.state_dict(),
        named_parameters=model.named_parameters(),
        named_buffers=model.named_buffers(),
    )
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # Ensure tmp directory exists
    os.makedirs("tmp", exist_ok=True)

    # Example usage
    text_list = ["walk in a circle.", "jump up."]
    text_end = [150, 250]
    length = text_end[-1]

    vae, model, cfg = load_model_from_config()
    feature_dim = vae.input_dim

    print("Starting generation...")
    # model.generate takes batched inputs; this is a single sample, so every
    # field is wrapped in a one-element list.
    text = [text_list]
    feature_text_end = [text_end]
    feature_length = [length]

    ik = model.input_keys
    x = {
        ik["feature_length"]: torch.tensor(feature_length),
        ik["text"]: text,
        ik["text_end"]: feature_text_end,
    }

    with torch.no_grad():
        # non-streaming generate
        print("Non-streaming generate...")
        torch.manual_seed(42)
        output = model.generate(x)
        nonstream_generated = output["generated"][0]  # (T, C) latent features
        print(f"Non-streaming generated shape: {nonstream_generated.shape}")
        nonstream_decoded = vae.decode(nonstream_generated[None, :])[0]  # (T_frames, feature_dim)
        print(f"Non-streaming decoded shape: {nonstream_decoded.shape}")
        render_single_video(
            motion=nonstream_decoded.cpu().numpy(),
            save_path="tmp/nonstream_generated.mp4",
            representation=cfg.representation,
        )
        print("Non-streaming generate done")

        # streaming generate step: model natives (init_generated +
        # stream_generate_step) accumulate latents one at a time; each new
        # latent is VAE stream-decoded in lockstep. The complete latent
        # sequence is then decoded in one shot for reference.
        print("Streaming generate...")
        vae.clear_cache()
        torch.manual_seed(42)
        # Model window (seq_len for init_generated): the stream path attends
        # to at most this many latents of history at each step.
        history_len = 100
        model.init_generated(history_len)
        text_end_with_zero = [0] + text_end
        durations = [
            t - b for t, b in zip(text_end_with_zero[1:], text_end_with_zero[:-1])
        ]

        stream_latents = []
        stream_features = []
        first_chunk = True
        for text_item, duration in zip(text_list, durations):
            for _ in range(duration):
                output = model.stream_generate_step({ik["text"]: [text_item]})
                g = output["generated"][0]
                if g.shape[0] == 0:
                    continue  # warmup: nothing committed yet
                stream_latents.append(g)
                decoded = vae.stream_decode(g[None, :], first_chunk=first_chunk)[0]
                first_chunk = False
                stream_features.append(decoded)
        vae.clear_cache()

        stream_latents = torch.cat(stream_latents, dim=0)    # (T, z)
        stream_features = torch.cat(stream_features, dim=0)  # (T_frames, feature_dim)
        print(f"Stream latents shape:  {stream_latents.shape}")
        print(f"Stream features shape: {stream_features.shape} (stream decode)")

        # full (non-stream) VAE decode of the complete stream latent sequence
        stream_full_decoded = vae.decode(stream_latents[None, :])[0]
        print(f"Full-decode features shape: {stream_full_decoded.shape}")

        # save the three final feature sequences
        np.save("tmp/nonstream_generated.npy", nonstream_decoded.cpu().numpy())
        np.save("tmp/stream_generated.npy", stream_features.cpu().numpy())
        np.save("tmp/stream_full_decoded.npy", stream_full_decoded.cpu().numpy())
        print("Saved features to tmp/{nonstream_generated,stream_generated,"
              "stream_full_decoded}.npy")

        # compare: same latents, stream decode vs full decode
        diff = (stream_features - stream_full_decoded).abs()
        mse = ((stream_features - stream_full_decoded) ** 2).mean(dim=-1)  # (T,)
        print("\n=== Stream decode vs full decode (same latents) ===")
        print(f"Overall MSE: {mse.mean():.8f} | MaxAbsDiff: {diff.max():.8f}")
        print(
            f"Per-frame MSE: first {mse[0]:.8f}, median {mse.median():.8f}, "
            f"max {mse.max():.8f} (frame {mse.argmax().item()})"
        )

        # compare: non-stream vs stream latents, frame by frame.
        # Theory: while the stream window still covers the full history
        # (latent index < history_len), both paths see the same context and
        # the error stays tiny; past history_len the stream window slides
        # away from the non-stream full context and the error grows.
        min_len = min(nonstream_generated.shape[0], stream_latents.shape[0])
        lat_mse = (
            (nonstream_generated[:min_len] - stream_latents[:min_len]) ** 2
        ).mean(dim=-1)
        pre = lat_mse[:history_len]
        post = lat_mse[history_len:min_len]
        print("\n=== Non-stream vs stream latents (per-latent MSE) ===")
        print(f"[0, {history_len}):    mean {pre.mean():.8f}  max {pre.max():.8f}")
        print(f"[{history_len}, {min_len}): mean {post.mean():.8f}  max {post.max():.8f}")
        for i in range(0, min_len, 25):
            print(f"  latent {i:4d}: MSE={lat_mse[i]:.8f}")

        render_single_video(
            motion=stream_features.cpu().numpy(),
            save_path="tmp/stream_generated.mp4",
            representation=cfg.representation,
        )
        print("Streaming generate done")
