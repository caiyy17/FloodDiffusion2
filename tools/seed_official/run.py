"""Generate, convert and evaluate SEED motions using the official Kimodo benchmark."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--kimodo-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kimodo-python', default=sys.executable, help='Python executable in the official Kimodo environment')
    parser.add_argument('--seed-data', type=Path)
    parser.add_argument('--llm2vec', type=Path, help='Optional local merged official LLM2Vec model')
    parser.add_argument('--text-line', type=int)
    parser.add_argument('--seed-offset', type=int, default=0)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--stages', nargs='+', choices=['generate', 'convert', 'embed', 'metrics'],
                        default=['generate', 'convert', 'embed', 'metrics'])
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    output = args.output.resolve()
    commands = {
        'generate': [sys.executable, str(here / 'generate.py'), '--config', str(args.config.resolve()),
                     '--suite', str(args.suite.resolve()), '--output', str(output),
                     '--seed-offset', str(args.seed_offset), '--limit', str(args.limit)],
        'convert': [args.kimodo_python, str(here / 'convert.py'), '--feats', str(output / 'feats'),
                    '--tree', str(output / 'tree'), '--suite', str(args.suite.resolve()),
                    '--kimodo-dir', str(args.kimodo_dir.resolve())],
        'embed': [args.kimodo_python, str(here / 'embed.py'), str(output / 'tree'),
                  '--kimodo-dir', str(args.kimodo_dir.resolve())],
        'metrics': [args.kimodo_python, str(here / 'metrics.py'), str(output / 'tree'),
                    '--kimodo-dir', str(args.kimodo_dir.resolve()), '--output', str(output)],
    }
    if args.seed_data:
        commands['generate'] += ['--seed-data', str(args.seed_data.resolve())]
    if args.text_line is not None:
        commands['generate'] += ['--text-line', str(args.text_line)]
    if args.llm2vec:
        commands['embed'] += ['--llm2vec', str(args.llm2vec.resolve())]
    for phase in ['generate', 'convert', 'embed', 'metrics']:
        if phase in args.stages:
            print(f'Running {phase}', flush=True)
            subprocess.run(commands[phase], check=True)


if __name__ == '__main__':
    main()
