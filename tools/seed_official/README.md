# SEED official evaluation

Evaluate the released text-only and path-conditioned SEED checkpoints with
the [Kimodo official benchmark](https://github.com/nv-tlabs/kimodo/tree/main/benchmark).
The adapter generates MEI138 features, converts them to SOMA77, then calls the
official TMR and physical metrics.

## Dependencies

Prepare the FloodDiffusion 2 environment, checkpoints and SEED data using the
project setup instructions. Install Kimodo and its TMR/LLM2Vec dependencies
following its official instructions, and obtain the
[SEED benchmark](https://huggingface.co/datasets/nvidia/Kimodo-Motion-Gen-Benchmark).
The pipeline was verified with Kimodo commit
`1aece8c124d73d255ceff5086d983b844c9f4e94` (v1.1 evaluation).

`--suite` points to `testsuite/content/text2motion/overview`, containing each
case's `meta.json`, `seed_motion.json` and `gt_motion.npz`.
Set `CHECKPOINT_DIR` to your Kimodo checkpoint directory. Kimodo can use a
separate Python environment via `--kimodo-python`.

## Run

From the FloodDiffusion 2 repository root:

```bash
export CHECKPOINT_DIR=/path/to/kimodo_checkpoints

python tools/seed_official/run.py \
  --config configs/df_seed_138.yaml \
  --suite /path/to/benchmark/testsuite/content/text2motion/overview \
  --kimodo-dir /path/to/kimodo \
  --kimodo-python /path/to/kimodo-env/bin/python \
  --output outputs/seed_official_text
```

For path control, use `configs/df_seed_138_path.yaml` and a separate output
directory. The checkpoint and CFG are read from the selected YAML. Choose a
GPU with `CUDA_VISIBLE_DEVICES` as usual.

Optional arguments:

- `--seed-data`: location of `SEED/MEI138`; otherwise read from `dirs.raw_data`.
- `--llm2vec`: local merged official LLM2Vec model, including tokenizer and
  `llm2vec_config.json`; otherwise use Kimodo's default loader.
- `--limit 2`: run a small pipeline check before the full evaluation.
- `--stages convert embed metrics`: evaluate already generated features.
- `--seed-offset`: add an offset to each official case seed for repeated runs.

## Protocol and outputs

Both configurations use the official case prompts. In the 917-case SEED
overview, these are identical to the fourth dataset captions. Path control
uses the official crop interval and the first three MEI138 channels.
`move_map.json` maps renamed benchmark motions to the released SEED IDs.
The tool checks that every selected path and caption is available before
loading the network. Durations follow the benchmark at 30 FPS.

Conversion recovers the 22 SMPL-H rotations and pelvis translation, maps
them to SOMA77, and reverses the SEED preprocessing scale of `0.915609`.
Kimodo's `complete_motion_dict` supplies the posed joints and velocity-based
contact labels. `benchmark/embed_folder.py` computes the embeddings, and
`compute_tmr_retrieval_metrics` computes retrieval and FID.

Outputs are `feats/`, `tree/`, `generation.json` and `metrics.json`.
The summary reports R@1/2/3 in percent, motion-to-GT FID (`TMR/FID/gen_gt`),
and contact-based foot skate in cm/s. The JSON also retains the full official
TMR and physical metric values. A missing case or failed conversion stops
the pipeline instead of changing the evaluation set.

Use a new output directory for each generation run. The tools retain the
generated features and benchmark tree for inspection.
