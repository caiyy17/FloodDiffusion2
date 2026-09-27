# Third-Party Code Attribution

This project uses code from the following open-source projects. All copyright notices are preserved in the source files.

---

## 1. Quaternion Operations

**File:** `visualization/HumanML3D263/quaternion.py`  
**Source:** Facebook Research - VideoPose3D  
**Repository:** https://github.com/facebookresearch/VideoPose3D  
**License:** BSD-3-Clause  
**Modifications:** Adaption for numpy

Copyright notice from file:

```
Copyright (c) 2018-present, Facebook, Inc. All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
```

---

## 2. Alibaba WAN Model Components

**Files:**

- `models/tools/t5.py`
- `models/tools/wan_model.py`
- `models/tools/wan_vae.py`
- `models/tools/attention.py`
- `models/tools/tokenizers.py`

**Source:** Alibaba Wan Team  
**Repository:** https://github.com/Wan-Video/Wan2.2
**License:** Apache 2.0  
**Modifications:** Modified for streaming motion generation (causal attention, stream mode, context length handling)

Copyright notice from files:

```
Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
```

**Note:** `t5.py` is also based on Hugging Face Transformers (Apache 2.0):

```
Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
Licensed under the Apache License, Version 2.0
```

---

## 3. Text-to-Motion Evaluation Metrics

**Files:**

- `metrics/HumanML3D263/t2m_evaluator.py`
- `metrics/HumanML3D263/word_vectorizer.py`
- `metrics/tools/utils.py`

**Source:** text-to-motion  
**Repository:** https://github.com/EricGuo5513/text-to-motion  
**License:** MIT  
**Modifications:** Adapted for this project's evaluation pipeline

These files implement standard text-to-motion metrics: R-precision, FID, diversity, and multimodality.

---

## 4. Motion Representations

### 4.1 HumanML3D (263-D)

**Files:**

- `visualization/HumanML3D263/*` (decode side: `quaternion.py`, recovery,
  skeleton rendering)
- `metrics/HumanML3D263/to_humanml3d.py`
- `metrics/tools/joints_to_humanml3d.py`

**Source:** HumanML3D (Guo et al., CVPR 2022)
**Repository:** https://github.com/EricGuo5513/HumanML3D
**License:** MIT (Copyright (c) 2022 Chuan Guo)
**Modifications:** approximate rot-channel -> SMPL retargeting in
`recovery.recover_smpl`; decode kept faithful to the official convention.

### 4.2 MotionStreamer (272-D)

**Files:**

- `visualization/MotionStreamer272/*`
- `metrics/MotionStreamer272/*`

**Source:** "MotionStreamer" / "Go to Zero" (ICCV 2025), Lixing Xiao
**Repository:** https://github.com/Li-xingXiao/272-dim-Motion-Representation
**License:** MIT (Copyright (c) 2025 Lixing Xiao)
**Modifications:** official recover_visualize.py decode kept verbatim apart
from renames.

### 4.3 NVIDIA kimodo / ardy (SOMA representation family)

**Files:**

- `visualization/SOMARelative271/*` (design reference: per-joint pos/rot6d/vel
  blocks, double-threshold foot contacts)
- `metrics/SOMARelative271/*`

**Source:** NVIDIA kimodo & ardy
**Repository:** https://github.com/nv-tlabs/kimodo, https://github.com/nv-tlabs/ardy
(project page: https://research.nvidia.com/labs/sil/projects/kimodo/)
**License:** Apache-2.0 (Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES)
**Modifications:** SE(2)-invariant redesign on the MEI-138 core (heading/root
increments instead of absolute root; heading-aligned pelvis-anchored positions);
kimodo double-threshold foot-contact heuristic kept.

### 4.4 MEI (138-D)

**Files:**

- `visualization/MEI138/*`
- `visualization/SOMARelative271/recovery.py` (inlined
  geometry helpers and the 138-core layout conventions)
- `visualization/tools/rotations.py` (quaternion/axis-angle utilities carried
  over from the upstream repo's geometry module, itself a numpy port of
  PyTorch3D rotation conversions, BSD-3)
- `visualization/tools/smpl_skeleton.py` (neutral rest joints baked from the
  upstream body model)

**Source:** MEI Representation (CAI YIYI)
**Repository:** https://github.com/caiyy17/MEI_Representation
**License:** MIT (Copyright (c) 2026 CAI YIYI)
**Modifications:** none in representation semantics — the upstream repo was
updated in lockstep (same 138 layout, raw per-frame heading, frame-diff
increments, column-concat 6D, de Leva CoM pivot). Local packaging differences
only: decode-side subset, neutral rest joints baked as constants instead of
loading the body-model file at runtime.

---

## Third-Party License Summary

Third-party components retain their original licenses. The Apache 2.0
license text is included in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).

| Component              | License      |
| ---------------------- | ------------ |
| Facebook quaternion.py | BSD-3-Clause |
| Alibaba WAN            | Apache 2.0   |
| HumanML3D metrics      | MIT          |
| HumanML3D 263-D rep    | MIT          |
| MotionStreamer 272-D   | MIT          |
| NVIDIA kimodo / ardy   | Apache 2.0   |
| MEI 138-D rep          | MIT          |

---

## Legal Notice

- The MIT License of this project applies **only** to original code by the FloodDiffusion 2 authors
- Third-party code remains under its original license
- All copyright notices are preserved in source files
- Modifications to third-party code are documented in file headers

For questions, open an issue in the project repository.

---
