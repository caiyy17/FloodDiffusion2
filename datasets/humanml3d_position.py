"""Position-conditioning dataset with the strict WAN-causal grouped-token ABI.

This module keeps every position-specific data contract out of the shared
``datasets/humanml3d.py``:

- ``representation`` selects the full feature width and root-motion prefix.
- ``data.token_dim`` / ``data.token_group_size`` declare the grouped-token ABI:
  ``[root_dim * group_size raw root values, latent]``, with the special
  one-frame first group.
- Token files must be finite real float32 2D arrays whose length matches the
  WAN-causal formula ``1 + (T - 1) // group_size`` of their source feature.
- Grouped tokens must never be window-cropped/time-sliced: token 0 covers only
  frame 0, so a cropped feature window has no valid token sub-slice.
"""

import os
import random
from typing import List

import numpy as np
from lightning.pytorch.utilities import rank_zero_info
from visualization import registry as representation_registry

from .humanml3d import (
    EmptyTextError,
    HumanML3DDataset,
    LengthMismatchError,
    collate_fn,  # noqa: F401  (re-exported for configs)
)


POSITION_ROOT_DIMS = {
    "humanml3d263": 3,
    "mei138": 3,
    "somarelative271": 3,
    "motionstreamer272": 8,
}
WAN_ROOT_GROUP_SIZE = 4


class TokenValidationError(Exception):
    """A tokenized sample is missing, unreadable, or violates its declared ABI."""


class HumanML3DPositionDataset(HumanML3DDataset):
    def __init__(self, cfg, split="train"):
        # These must exist before super().__init__ because the base constructor
        # runs _load_file_list(), which dispatches to the overrides below.
        representation = cfg.get("representation", None)
        if representation is None:
            representation = cfg.data.get("representation", "mei138")
        self.representation = representation_registry.canonical(representation)
        if self.representation not in POSITION_ROOT_DIMS:
            raise ValueError(
                f"representation {self.representation!r} has no position root "
                f"contract; supported: {sorted(POSITION_ROOT_DIMS)}"
            )
        self.feature_dim = representation_registry.dim(self.representation)
        self.root_dim = POSITION_ROOT_DIMS[self.representation]
        self.token_dim = cfg.data.get("token_dim", None)
        self.token_group_size = cfg.data.get("token_group_size", None)
        # The plain token datasets use uniform feature/token FPS slicing.  The
        # explicit grouped-token ABI is WAN-causal and has a special first token,
        # so it must use the stricter no-window-crop contract below.
        self.strict_grouped_token_abi = self.token_group_size is not None
        if self.token_dim is not None and int(self.token_dim) <= 0:
            raise ValueError(f"data.token_dim must be positive, got {self.token_dim}")
        if self.token_group_size is not None:
            if self.token_dim is None:
                raise ValueError(
                    "data.token_group_size enables the strict grouped-token ABI "
                    "and therefore requires data.token_dim"
                )
            self.token_dim = int(self.token_dim)
            self.token_group_size = int(self.token_group_size)
            if self.token_group_size != WAN_ROOT_GROUP_SIZE:
                raise ValueError(
                    "the WAN grouped-token ABI requires "
                    f"data.token_group_size={WAN_ROOT_GROUP_SIZE}, got "
                    f"{self.token_group_size}"
                )
            self.position_dim = self.root_dim * self.token_group_size
            if self.token_dim <= self.position_dim:
                raise ValueError(
                    f"data.token_dim must exceed the {self.position_dim}D packed "
                    f"root condition for {self.representation}, got {self.token_dim}"
                )
            if cfg.data.get("token_path", None) is None or (
                cfg.data.get("feature_path", None) is None
            ):
                raise ValueError(
                    "data.token_group_size requires both data.token_path and "
                    "data.feature_path"
                )
        else:
            self.position_dim = None
        super().__init__(cfg, split=split)

    def _expected_packed_root(self, feature, token_length):
        """Build the raw WAN-causal root prefix expected in saved tokens."""
        roots = feature[:, : self.root_dim]
        token_ids = np.arange(int(token_length), dtype=np.int64)[:, None]
        offsets = np.arange(self.token_group_size, dtype=np.int64)[None, :]
        indices = 1 + (token_ids - 1) * self.token_group_size + offsets
        indices[0] = 0
        indices = np.clip(indices, 0, len(roots) - 1)
        return roots[indices].reshape(int(token_length), self.position_dim)

    def _load_file_list(self) -> List[str]:
        if (
            self.split == "train"
            and self.strict_grouped_token_abi
            and self.window_length < self.max_length
        ):
            raise ValueError(
                "window-cropping pretokenized motions is not WAN-causally aligned; "
                "set data.window_length >= data.max_length or re-encode each cropped "
                "feature window instead of slicing full-clip tokens"
            )
        if self.strict_grouped_token_abi:
            missing_manifests = [
                (e if isinstance(e, str) else e["path"])
                for e in self.file_list
                if not os.path.isfile(e if isinstance(e, str) else e["path"])
            ]
            if missing_manifests:
                raise FileNotFoundError(
                    f"grouped-token manifest(s) not found: {missing_manifests}"
                )
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
                                if self.feature_path is not None:
                                    feature_path = os.path.join(
                                        data_path, self.feature_path, name + ".npy"
                                    )
                                    feature = self.load_feature(feature_path)
                                    data["feature"] = feature
                                    data["feature_length"] = feature.shape[0]
                                if self.token_path is not None:
                                    token_path = os.path.join(
                                        data_path,
                                        self.token_path,
                                        name + ".npy",
                                    )
                                    token = self.load_token(token_path)
                                    if self.token_group_size is not None:
                                        expected_token_length = 1 + (
                                            data["feature_length"] - 1
                                        ) // self.token_group_size
                                        if token.shape[0] != expected_token_length:
                                            raise TokenValidationError(
                                                f"token length {token.shape[0]} does "
                                                f"not match WAN-causal feature length "
                                                f"{data['feature_length']} (expected "
                                                f"{expected_token_length})"
                                            )
                                        expected_root = self._expected_packed_root(
                                            data["feature"], token.shape[0]
                                        )
                                        if not np.allclose(
                                            token[:, : self.position_dim],
                                            expected_root,
                                            rtol=0.0,
                                            atol=1e-6,
                                        ):
                                            max_error = float(
                                                np.max(
                                                    np.abs(
                                                        token[
                                                            :, : self.position_dim
                                                        ].astype(np.float64)
                                                        - expected_root.astype(
                                                            np.float64
                                                        )
                                                    )
                                                )
                                            )
                                            raise TokenValidationError(
                                                "token root prefix does not match "
                                                f"{self.representation} source root "
                                                f"packing (max_abs={max_error})"
                                            )
                                    data["token"] = token
                                    data["token_length"] = token.shape[0]
                                if self.text_path is not None:
                                    text_path = os.path.join(
                                        data_path, self.text_path, name + ".txt"
                                    )
                                    text_data = self.load_text(text_path)
                                    data["text_data"] = text_data
                                dataset.append(data)
                            except LengthMismatchError:
                                ignored_cnt += 1
                            except EmptyTextError:
                                empty_text_cnt += 1
                            except TokenValidationError as e:
                                raise RuntimeError(
                                    f"Invalid token data for {name}: {e}"
                                ) from e
                            except Exception as e:
                                if self.token_path is not None:
                                    raise RuntimeError(
                                        f"Error loading tokenized data for {name}: {e}"
                                    ) from e
                                rank_zero_info(f"Error loading data for {name}: {e}")
                            if self.cfg.debug and len(dataset) >= 100:
                                rank_zero_info(f"debug mode, break at {len(dataset)}")
                                break
        if ignored_cnt > 0:
            rank_zero_info(f"Ignored {ignored_cnt} samples due to length mismatch.")
        if empty_text_cnt > 0:
            rank_zero_info(f"Ignored {empty_text_cnt} samples due to empty text files.")
        if len(dataset) == 0:
            rank_zero_info(
                f"No data found in {self.file_list}. Please check the file paths "
                "and ensure they are correct."
            )
        else:
            for i in range(3):
                tmp = random.choice(dataset)
                rank_zero_info(f"Random data {tmp['name']}: {tmp['feature'].shape}")
        return dataset

    def load_feature(self, feature_path: str) -> np.ndarray:
        feature = np.load(feature_path, allow_pickle=False)
        if self.strict_grouped_token_abi:
            if (
                not isinstance(feature, np.ndarray)
                or feature.ndim != 2
                or feature.shape[0] < 1
                or feature.shape[1] != self.feature_dim
                or not np.issubdtype(feature.dtype, np.number)
                or np.issubdtype(feature.dtype, np.complexfloating)
                or not np.isfinite(feature).all()
            ):
                raise ValueError(
                    "grouped-token source feature must be finite real "
                    f"(T,{self.feature_dim}) for {self.representation}, "
                    f"got {getattr(feature, 'shape', None)}"
                    f"/{getattr(feature, 'dtype', None)}"
                )
        feature = feature.astype(np.float32)
        if not np.isfinite(feature).all():
            raise ValueError("NaN/Inf values found in feature, skip it.")
        if feature.shape[0] < self.min_length or feature.shape[0] > self.max_length:
            raise LengthMismatchError("Feature length out of bounds, skip it.")
        return feature

    def load_token(self, token_path: str) -> np.ndarray:
        try:
            token = np.load(token_path)
        except Exception as e:
            raise TokenValidationError(
                f"could not load token file {token_path}: {e}"
            ) from e
        if not isinstance(token, np.ndarray) or token.ndim != 2 or token.shape[0] < 1:
            raise TokenValidationError(
                f"expected a non-empty 2D token array, got {getattr(token, 'shape', None)}"
            )
        if not np.issubdtype(token.dtype, np.number) or np.issubdtype(
            token.dtype, np.complexfloating
        ):
            raise TokenValidationError(
                f"expected real numeric token dtype, got {token.dtype}"
            )
        if not np.isfinite(token).all():
            raise TokenValidationError("token contains NaN/Inf")
        if self.strict_grouped_token_abi and token.dtype != np.float32:
            raise TokenValidationError(
                f"grouped-token ABI requires float32, got {token.dtype}"
            )
        if (
            self.strict_grouped_token_abi
            and token.shape[1] <= self.position_dim
        ):
            raise TokenValidationError(
                f"token width {token.shape[1]} must exceed the packed root "
                f"condition width {self.position_dim}"
            )
        if self.token_dim is not None and token.shape[1] != int(self.token_dim):
            raise TokenValidationError(
                f"token width {token.shape[1]} does not match declared "
                f"data.token_dim={int(self.token_dim)}"
            )
        return token

    def _process(self, data):
        if (
            self.strict_grouped_token_abi
            and "feature" in data
            and data["feature"].shape[0] > self.window_length
        ):
            raise RuntimeError(
                "refusing to time-slice a pretokenized motion after feature "
                "window-cropping because token 0 has a special one-frame WAN "
                "causal group; re-encode the cropped feature window"
            )
        return super()._process(data)
