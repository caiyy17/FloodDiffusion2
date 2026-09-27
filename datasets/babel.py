import math
import os
import random
from typing import Dict, List

import numpy as np
import torch
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.initialize import instantiate


class LengthMismatchError(Exception):
    pass


class EmptyTextError(Exception):
    pass


class BabelDataset(Dataset):
    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        self.split = split
        if self.split == "train":
            self.file_list = cfg.data.train_meta_paths
            self.min_length = cfg.data.min_length
            self.max_length = cfg.data.max_length
            self.window_length = cfg.data.get("window_length", cfg.data.max_length)
            self.random_length = cfg.data.get("random_length", 0)
        elif self.split == "val":
            self.file_list = cfg.data.val_meta_paths
            self.min_length = cfg.data.min_length
            self.max_length = cfg.data.max_length
            self.window_length = cfg.data.max_length
            self.random_length = 0
        elif self.split == "test":
            self.file_list = cfg.data.test_meta_paths
            self.min_length = 0
            self.max_length = cfg.data.max_length
            self.window_length = cfg.data.max_length
            self.random_length = 0
        self.feature_path = cfg.data.get("feature_path", None)
        self.token_path = cfg.data.get("token_path", None)
        self.text_path = cfg.data.get("text_path", None)
        self.dataset = self._load_file_list()

        rank_zero_info(f"Loaded {len(self.dataset)} samples from {split} dataset.")

    def _load_file_list(self) -> List[str]:
        """Load a list of npy file paths from a text file."""
        dataset = []
        ignored_cnt = 0
        empty_text_cnt = 0
        for entry in self.file_list:
            # each entry is a mapping {path: ..., name: ...}; the explicit
            # dataset name drives eval grouping / test_setting keys.
            try:
                path, dataset_name = entry["path"], entry["name"]
            except (TypeError, KeyError) as exc:      # str entry / missing key
                raise ValueError(
                    f"meta entry must be a mapping {{path, name}}, got {entry!r}"
                ) from exc
            if os.path.exists(path):
                data_path = os.path.dirname(path)
                rank_zero_info(f"Loading {path} (dataset={dataset_name}) ...")
                with open(path, "r") as f:
                    for name in f:
                        name = name.strip()
                        if name:
                            data = {}
                            try:
                                data["name"] = name
                                data["dataset"] = dataset_name
                                ##############################
                                # feature
                                ##############################
                                if self.feature_path is not None:
                                    feature_path = os.path.join(
                                        data_path, self.feature_path, name + ".npy"
                                    )
                                    feature = self.load_feature(feature_path)
                                    data["feature"] = feature
                                    data["feature_length"] = feature.shape[0]
                                ##############################
                                # token
                                ##############################
                                if self.token_path is not None:
                                    token_path = os.path.join(
                                        data_path,
                                        self.token_path,
                                        name + ".npy",
                                    )
                                    token = self.load_token(token_path)
                                    data["token"] = token
                                    data["token_length"] = token.shape[0]
                                ##############################
                                # text
                                ##############################
                                if self.text_path is not None:
                                    text_path = os.path.join(
                                        data_path, self.text_path, name + ".txt"
                                    )
                                    text_data = self.load_text(text_path)
                                    data["text_data"] = text_data
                                ##############################
                                # end
                                ##############################
                                dataset.append(data)
                            except LengthMismatchError:
                                ignored_cnt += 1
                                pass
                            except EmptyTextError:
                                empty_text_cnt += 1
                                pass
                            except Exception as e:
                                rank_zero_info(f"Error loading data for {name}: {e}")
                                pass
                            if self.cfg.debug and len(dataset) >= 100:
                                rank_zero_info(f"debug mode, break at {len(dataset)}")
                                break
        if ignored_cnt > 0:
            rank_zero_info(f"Ignored {ignored_cnt} samples due to length mismatch.")
        if empty_text_cnt > 0:
            rank_zero_info(f"Ignored {empty_text_cnt} samples due to empty text files.")
        if len(dataset) == 0:
            rank_zero_info(
                f"No data found in {self.file_list}. Please check the file paths and ensure they are correct."
            )
        else:
            for i in range(3):
                tmp = random.choice(dataset)
                rank_zero_info(f"Random data {tmp['name']}: {tmp['feature'].shape}")
        return dataset

    def load_feature(self, feature_path: str) -> np.ndarray:
        feature = np.load(feature_path).astype(np.float32)
        if np.isnan(feature).any():
            raise ValueError("NaN values found in feature, skip it.")
        if feature.shape[0] < self.min_length or feature.shape[0] > self.max_length:
            raise LengthMismatchError("Feature length out of bounds, skip it.")
        return feature

    def load_token(self, token_path: str) -> np.ndarray:
        token = np.load(token_path)
        if np.isnan(token).any():
            raise ValueError("NaN values found in token, skip it.")
        return token

    def load_text(self, text_path: str) -> List[Dict]:
        text_data = []
        with open(text_path, "r") as text_f:
            lines = text_f.readlines()
            for line in lines:
                text_dict = {}
                line_split = line.strip().split("#")
                caption = line_split[0]
                t_tokens = line_split[1].split(" ")
                f_tag = float(line_split[2])
                to_tag = float(line_split[3])
                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                to_tag = 0.0 if np.isnan(to_tag) else to_tag

                text_dict["caption"] = caption
                text_dict["tokens"] = t_tokens
                text_dict["f_tag"] = f_tag
                text_dict["to_tag"] = to_tag

                text_data.append(text_dict)
        if not text_data:
            raise EmptyTextError(f"Empty text file (no parseable lines): {text_path}")
        return text_data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        return self._process(data)

    def _process(self, data):
        output = {}
        output["dataset"] = data["dataset"]
        output["name"] = data["name"]

        feature_fps = self.cfg.data.feature_fps
        token_fps = self.cfg.data.token_fps

        # Crop window is defined in FEATURE space (window_length is a feature length).
        # Pick a feature window, express it as a [start, end] TIME range (seconds), then
        # derive the token window and crop the text segments from that same time range.
        # This aligns feature / token / text via TIME (no hardcoded downsample factor) and
        # works even without tokens (raw-feature training) since the window comes from feature.
        feature_start = 0
        crop_start_sec = 0.0
        crop_end_sec = self.window_length / feature_fps
        cropped = False
        if "feature" in data:
            feature = data["feature"]
            feature_len = feature.shape[0]
            if feature_len > self.window_length:
                feature_start = random.randint(0, feature_len - self.window_length)
                cropped = True
            feature_end = min(feature_start + self.window_length, feature_len)
            feature = feature[feature_start:feature_end]
            output["feature"] = feature
            output["feature_length"] = feature.shape[0]
            crop_start_sec = feature_start / feature_fps
            crop_end_sec = feature_end / feature_fps

        # Derive the token window from the feature window's time range; if the feature was
        # NOT windowed, use the whole token (matches non-windowed behavior, no 4k+1 drop).
        if "token" in data:
            token = data["token"]
            if cropped:
                token_start = math.floor(crop_start_sec * token_fps + 1 / feature_fps)
                token_end = math.floor(crop_end_sec * token_fps + 1 / feature_fps)
                token = token[token_start:token_end]
            output["token"] = token
            output["token_length"] = token.shape[0]

        # Crop and adjust text annotations by the same time range.
        if "text_data" in data:
            output["text"] = []
            output["text_tokens"] = []
            output["feature_text_end"] = []
            output["token_text_end"] = []

            for text_dict in data["text_data"]:
                text, text_tokens, f_tag, to_tag = self.process_text_dict(text_dict)
                # Skip segments entirely outside the crop window
                if to_tag <= crop_start_sec or f_tag >= crop_end_sec:
                    continue
                # Adjust to_tag relative to crop start, clamp to crop window
                adjusted_to = min(to_tag, crop_end_sec) - crop_start_sec
                feature_text_end = math.floor(adjusted_to * feature_fps + 1 / feature_fps)
                token_text_end = math.floor(adjusted_to * token_fps + 1 / token_fps)
                # Skip segments that round to 0 frames (barely overlapping)
                if feature_text_end <= 0:
                    continue
                output["text"].append(text)
                output["text_tokens"].append(text_tokens)
                output["feature_text_end"].append(feature_text_end)
                output["token_text_end"].append(token_text_end)
            # Fallback: no text overlaps with crop window
            if len(output["text"]) == 0:
                output["text"] = [""]
                output["text_tokens"] = [[""]]
                output["feature_text_end"] = [output.get("feature_length", 0)]
                output["token_text_end"] = [output.get("token_length", 0)]

            # concat every segment's word tokens (in temporal order) into one flat
            # list -> a full-motion description, instead of keeping only the last segment.
            output["text_tokens"] = [
                tok for seg_tokens in output["text_tokens"] for tok in seg_tokens
            ]

            # Force the last boundary to the actual window length so the last segment
            # always spans to the window end (covers SEED texts ending a few frames
            # early + any rounding gap).
            output["feature_text_end"][-1] = output.get("feature_length", 0)
            output["token_text_end"][-1] = output.get("token_length", 0)
        return output

    def process_text_dict(self, text_dict: Dict):
        text = text_dict["caption"]
        text_tokens = text_dict["tokens"]
        f_tag = text_dict["f_tag"]
        to_tag = text_dict["to_tag"]
        return text, text_tokens, f_tag, to_tag


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    output = {}
    keys = batch[0].keys()

    for key in keys:
        if key in ["feature", "token"]:
            # Pad sequences
            items = [
                torch.from_numpy(b[key]) if isinstance(b[key], np.ndarray) else b[key]
                for b in batch
            ]
            output[key] = torch.nn.utils.rnn.pad_sequence(
                items, batch_first=True, padding_value=0
            )
        elif key in ["feature_length", "token_length"]:
            # Stack scalars
            output[key] = torch.tensor([b[key] for b in batch])
        else:
            # Default to list
            output[key] = [b[key] for b in batch]
    return output


if __name__ == "__main__":
    cfg = OmegaConf.load("configs/default.yaml")
    dataset = BabelDataset(cfg, split="val")
    print(len(dataset))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.data.train_bs, shuffle=True, collate_fn=collate_fn
    )
    for idx, data in tqdm(enumerate(dataloader)):
        pass
