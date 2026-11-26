"""Measure how perturbation width and direction count affect decision feedback."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gnn_bdge.data import load_onet_profiles, make_market, procedural_profiles, save_metadata
from gnn_bdge.training import fit_models
from gnn_bdge.solvers import solve_highs, utc_now
from gnn_bdge.experiments import write_csv, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/quick.json')
    parser.add_argument('--results', default='output/spsa_sensitivity')
    parser.add_argument('--source-mode', choices=['onet31', 'procedural'], default='onet31')
    parser.add_argument('--onet')
    parser.add_argument('--N', type=int, default=200)
    parser.add_argument('--seeds', nargs='+', type=int, default=[41,73,107,149,181])
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.source_mode == 'onet31':
        profiles, source = load_onet_profiles(args.onet, config['skill_dimensions'])
    else:
        profiles, source = procedural_profiles(config['profile_seed'], config['skill_dimensions'])
    kwargs = dict(n=args.N, k=config['skill_dimensions'],capacity_mean=config['capacity_mean'],
                  tau=config['candidate_threshold'],noise_std=config['worker_noise'],
                  required_skills=config['required_skills'],minimum_ratio=config['qualification_ratio'])
    cases = [(p, 0.0001) for p in (1,5,10,20)] + [(10, delta) for delta in (1e-5,1e-3)]
    records = []
    for seed in args.seeds:
        training = [make_market(profiles, seed=seed*1000+i, **kwargs) for i in range(7)]
        validation = make_market(profiles, seed=seed*1000+7, **kwargs)
        for p, delta in cases:
            current = dict(config)
            current['spsa_directions'] = p
            current['spsa_delta'] = delta
            current['spsa_steps'] = 1
            _, _, trace = fit_models(training, validation, current, seed)
            step = trace['gradient_trace'][0]
            records.append({'N':args.N,'seed':seed,'P':p,'delta':delta,
                            'gradient_norm':step['gradient_norm'],
                            'decision_loss':step['decision_loss_after_update'],
                            'mean_direction_derivative':step['difference_mean'],
                            'training_seconds':trace['train_wall_seconds'],
                            'source_mode':args.source_mode,'timestamp_utc':utc_now()})
        print(f'SPSA sweep completed for seed {seed}', flush=True)
    output = Path(args.results)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'spsa_parameter_sweep.csv', records)
    summary = []
    for p, delta in cases:
        group = [r for r in records if r['P']==p and r['delta']==delta]
        entry = {'N':args.N,'P':p,'delta':delta,'completed_seeds':len(group)}
        for key in ('gradient_norm','decision_loss','mean_direction_derivative','training_seconds'):
            entry[key+'_mean'],entry[key+'_sd'] = stats([r[key] for r in group])
        summary.append(entry)
    write_csv(output/'spsa_parameter_summary.csv', summary)
    save_metadata(output/'run_details.json', {'created_at_utc':utc_now(), 'source':source,
                   'config':config,'cases': [{'P':p,'delta':d} for p,d in cases]})
    print(f'Parameter sweep: {len(records)} runs')


if __name__ == '__main__':
    main()
