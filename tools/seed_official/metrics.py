"""Aggregate official TMR embeddings and evaluate official SOMA physical metrics."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np


def vector(path):
    value = np.load(path)
    while value.ndim > 1 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError(f'Invalid embedding: {path}')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--kimodo-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.kimodo_dir.resolve()))
    import torch
    from kimodo.metrics.tmr import compute_tmr_retrieval_metrics
    from kimodo.metrics import (FootContactConsistency, FootSkateFromContacts,
                               FootSkateFromHeight, FootSkateRatio, aggregate_metrics, compute_metrics)
    from kimodo.skeleton import SOMASkeleton77

    torch.set_num_threads(4)
    folders = sorted(p.parent for p in args.folder.rglob('motion.npz'))
    if not folders:
        raise ValueError(f'No benchmark motions found in {args.folder}')
    motion, text, gt = [np.stack([vector(p / filename) for p in folders]) for filename in
                       ['motion_embedding.npy', 'text_embedding.npy', 'gt_motion_embedding.npy']]
    tmr = compute_tmr_retrieval_metrics(motion, text, gt_motion_emb=gt, rounding=4)
    skeleton = SOMASkeleton77().cpu()
    measures = [FootSkateFromHeight(skeleton, 30.0), FootSkateFromContacts(skeleton, 30.0),
                FootSkateRatio(skeleton, 30.0), FootContactConsistency(skeleton, 30.0)]
    frames = 0
    for folder in folders:
        with np.load(folder / 'motion.npz', allow_pickle=False) as data:
            joints = torch.from_numpy(data['posed_joints']).float()
            contacts = torch.from_numpy(data['foot_contacts']).bool()
        if joints.shape[1:] != (77, 3) or len(contacts) != len(joints) or not torch.isfinite(joints).all():
            raise ValueError(f'Invalid converted SOMA77 motion: {folder}')
        compute_metrics(measures, {'posed_joints': joints, 'foot_contacts': contacts,
                                   'lengths': torch.tensor(len(joints), dtype=torch.long)})
        frames += len(joints)
    physical = {key: float(value.float().mean()) for key, value in aggregate_metrics(measures).items()}
    result = {'protocol': 'Kimodo v1.1 official TMR and physical metrics', 'motions': len(folders),
              'frames': frames, 'tmr': tmr, 'physical_si': physical,
              'summary': {'R1_percent': tmr['TMR/t2m_R/R01'], 'R2_percent': tmr['TMR/t2m_R/R02'],
                          'R3_percent': tmr['TMR/t2m_R/R03'], 'FID': tmr['TMR/FID/gen_gt'],
                          'Skate_cm_per_s': 100.0 * physical['foot_skate_from_pred_contacts']}}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
