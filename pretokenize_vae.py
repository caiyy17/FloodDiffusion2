"""Compatibility entry for the config-driven VAE tokenizer.

Accepts the same arguments as pretokenize_flood2_vae.py. Dataset locations,
checkpoint, output directory, and training statistics split are explicit.
"""

from pretokenize_flood2_vae import main


if __name__ == "__main__":
    main()
