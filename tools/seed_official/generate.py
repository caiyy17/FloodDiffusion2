"""Generate SEED benchmark cases with a released FloodDiffusion 2 config."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def load_cases(suite, seed_data, move_map, path_control, text_line, limit=0):
    import numpy as np

    cases = []
    for folder in sorted(suite.iterdir()):
        if not (folder / 'meta.json').is_file() or not (folder / 'seed_motion.json').is_file():
            continue
        meta = json.loads((folder / 'meta.json').read_text())
        motion = json.loads((folder / 'seed_motion.json').read_text())
        frames = max(round(float(meta['duration']) * 30), 30)
        move = move_map.get(motion['move_name'], motion['move_name'])
        prompt = meta['text']
        position = None
        if text_line >= 0:
            lines = [line.split('#')[0].strip() for line in
                     (seed_data / 'texts' / (move + '.txt')).read_text().splitlines() if line.strip()]
            if len(lines) <= text_line:
                raise ValueError(f'{folder.name}: missing caption line {text_line}')
            prompt = lines[text_line]
        if path_control:
            feature = np.load(seed_data / 'new_joint_vecs_uni' / (move + '.npy'), mmap_mode='r')
            start = int(motion['crop_start_frame_index'])
            if start < 0 or start + frames > len(feature):
                raise ValueError(f'{folder.name}: unavailable official crop [{start}, {start + frames})')
            position = np.array(feature[start:start + frames, :3], dtype=np.float32, copy=True)
        cases.append({'id': folder.name, 'frames': frames, 'seed': int(meta['seed']),
                      'text': prompt, 'position': position})
        if limit and len(cases) >= limit:
            break
    if not cases:
        raise ValueError(f'No benchmark cases found in {suite}')
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--suite', type=Path, required=True, help='Official content/text2motion/overview directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed-data', type=Path, help='SEED/MEI138; defaults to the configured dataset')
    parser.add_argument('--move-map', type=Path, default=Path(__file__).with_name('move_map.json'))
    parser.add_argument('--text-line', type=int, default=-1, help='Zero-based dataset caption index; default -1 selects official text')
    parser.add_argument('--seed-offset', type=int, default=0)
    parser.add_argument('--limit', type=int, default=0, help='Generate only the first N cases for a smoke test')
    args = parser.parse_args()
    args.config, args.suite, args.output = args.config.resolve(), args.suite.resolve(), args.output.resolve()
    if args.seed_data:
        args.seed_data = args.seed_data.resolve()
    move_map = json.loads(args.move_map.read_text())
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    from utils.initialize import Config
    from generate_ldf import load_model_from_config

    cfg = Config(str(args.config)).config
    if cfg.representation != 'mei138' or int(cfg.model.params.input_dim) != 138:
        raise ValueError('This adapter requires a SEED MEI138 diffusion configuration')
    path_control = cfg.model.target == 'models.diffusion_forcing_position_wan.DiffForcingPositionWanModel'
    text_line = args.text_line
    seed_data = args.seed_data or Path(cfg.dirs.raw_data) / 'SEED/MEI138'
    cases = load_cases(args.suite, seed_data, move_map, path_control, text_line, args.limit)
    feats_dir = args.output / 'feats'
    if feats_dir.exists() and any(feats_dir.glob('*.npy')):
        raise FileExistsError(f'Use a new output directory; generated features already exist: {feats_dir}')
    feats_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'cases': [x['id'] for x in cases], 'config': str(args.config),
                'checkpoint': str(cfg.test_ckpt), 'path_control': path_control,
                'text_line': text_line, 'seed_offset': args.seed_offset,
                'cfg': float(cfg.model.params.cfg_config.text_scale), 'completed': 0}
    manifest_path = args.output / 'generation.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    sys.argv = ['generate', '--config', str(args.config)]
    _, model, _ = load_model_from_config()
    ik = model.input_keys
    for index, case in enumerate(cases):
        frames = case['frames']
        batch = {ik['feature_length']: torch.tensor([frames]),
                 ik['text']: [[case['text']]], ik['text_end']: [[frames]]}
        if path_control:
            batch['position'] = [torch.from_numpy(case['position'])]
        torch.manual_seed(case['seed'] + args.seed_offset)
        with torch.no_grad():
            features = model.generate(batch)['generated'][0][:frames].float().cpu().numpy()
        if features.shape != (frames, 138) or not np.isfinite(features).all():
            raise ValueError(f"Invalid generated motion: {case['id']}")
        np.save(feats_dir / (case['id'] + '.npy'), features.astype(np.float32))
        manifest['completed'] = index + 1
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        if (index + 1) % 25 == 0 or index + 1 == len(cases):
            print(f'Generated {index + 1}/{len(cases)}', flush=True)


if __name__ == '__main__':
    main()
