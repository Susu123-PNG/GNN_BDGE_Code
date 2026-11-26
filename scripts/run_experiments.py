"""Command-line entry point for market matching experiments."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gnn_bdge.experiments import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/standard.json')
    parser.add_argument('--results', default='output/standard')
    parser.add_argument('--source-mode', choices=['onet31', 'procedural'], default='onet31')
    parser.add_argument('--onet', help='Path to O*NET 31.0 CSV archive or extracted directory')
    parser.add_argument('--esco', help='Path to ESCO 1.2.1 extracted CSV directory')
    parser.add_argument('--gurobi', action='store_true')
    parser.add_argument('--utility-mode', choices=['structural', 'independent'])
    parser.add_argument('--sizes', nargs='+', type=int)
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--train-epochs', type=int)
    parser.add_argument('--spsa-steps', type=int)
    parser.add_argument('--spsa-directions', type=int)
    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as file:
        config = json.load(file)
    for key in ('sizes', 'seeds', 'train_epochs', 'spsa_steps', 'spsa_directions'):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.utility_mode is not None:
        config['utility_mode'] = args.utility_mode
    run(config, args.results, args.onet, args.esco, args.source_mode, args.gurobi)


if __name__ == '__main__':
    main()
