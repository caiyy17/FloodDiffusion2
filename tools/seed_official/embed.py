#!/usr/bin/env python3
"""Run the official Kimodo embedder, optionally using local merged LLM2Vec weights."""

import argparse
import json
import os
import runpy
import sys
from pathlib import Path

import torch


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('folder', type=Path)
parser.add_argument('--kimodo-dir', type=Path, required=True)
parser.add_argument('--llm2vec', type=Path)
args = parser.parse_args()
KIMODO = args.kimodo_dir.resolve()
MERGED = args.llm2vec.resolve() if args.llm2vec else None
sys.path.insert(0, str(KIMODO))
os.environ["TEXT_ENCODER_MODE"] = "local"


def patch_local_text_encoder() -> None:
    from kimodo.model.llm2vec import llm2vec_wrapper
    from kimodo.model.llm2vec.llm2vec import LLM2Vec
    from kimodo.model.llm2vec.models.bidirectional_llama import LlamaBiModel
    from transformers import AutoTokenizer

    def local_init(
        self, base_model_name_or_path, peft_model_name_or_path, dtype, llm_dim,
        device="auto",
    ):
        self.llm_dim = llm_dim
        tokenizer = AutoTokenizer.from_pretrained(MERGED)
        model = LlamaBiModel.from_pretrained(MERGED, torch_dtype=getattr(torch, dtype))
        model.config._name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
        cfg = json.loads((MERGED / "llm2vec_config.json").read_text())
        self.model = LLM2Vec(model=model, tokenizer=tokenizer, **cfg)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self.model = self.model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    llm2vec_wrapper.LLM2VecEncoder.__init__ = local_init


if MERGED is not None:
    patch_local_text_encoder()
sys.argv = [str(KIMODO / 'benchmark/embed_folder.py'), str(args.folder.resolve()), '--overwrite']
runpy.run_path(str(KIMODO / "benchmark/embed_folder.py"), run_name="__main__")
