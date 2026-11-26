"""Evaluate utility and capacity changes on independent held-out markets."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gnn_bdge.data import load_onet_profiles, procedural_profiles, make_market, save_metadata
from gnn_bdge.training import fit_models
from gnn_bdge.solvers import solve_highs, greedy, validate, utc_now
from gnn_bdge.experiments import write_csv, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/quick.json')
    parser.add_argument('--source-mode', choices=['onet31', 'procedural'], default='onet31')
    parser.add_argument('--onet')
    parser.add_argument('--results', default='output/robustness')
    parser.add_argument('--N', type=int, default=200)
    parser.add_argument('--seeds', type=int, nargs='+', default=[41,73,107,149,181])
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    output = Path(args.results)
    output.mkdir(parents=True, exist_ok=True)
    if args.source_mode == 'onet31':
        profiles, source_info = load_onet_profiles(args.onet, config['skill_dimensions'])
    else:
        profiles, source_info = procedural_profiles(config['profile_seed'], config['skill_dimensions'])
    kwargs = dict(n=args.N,k=config['skill_dimensions'],capacity_mean=config['capacity_mean'],
                  tau=config['candidate_threshold'],noise_std=config['worker_noise'],
                  required_skills=config['required_skills'],minimum_ratio=config['qualification_ratio'])
    rows = []
    for seed in args.seeds:
        train = [make_market(profiles, seed=seed*1000 + i, **kwargs) for i in range(7)]
        val = make_market(profiles, seed=seed*1000 + 7, **kwargs)
        model, _, _ = fit_models(train, val, config, seed)
        model.eval()
        for shift in (0., .10, .20, .30, .40):
            market = make_market(profiles, seed=seed*1000+30, shift=shift, **kwargs)
            oracle = solve_highs(market, market['utility'], 'mip', time_limit=config['solver_time_limit'],
                                 log_file=output/f'HiGHS_seed{seed}_shift{int(100*shift)}.jsonl')
            if not oracle['feasible']:
                continue
            t0 = time.perf_counter()
            with torch.no_grad():
                predicted = model(market).cpu().numpy().astype(float)
            infer = time.perf_counter()-t0
            t0 = time.perf_counter()
            chosen = greedy(market, predicted)
            solver = time.perf_counter()-t0
            actual = float(market['utility'][chosen].sum())
            rows.append({'N':args.N,'seed':seed,'shift_pct':int(100*shift),
                         'source_mode':args.source_mode,'utility_mode':'structural',
                         'capacity':market['total_capacity'],'B':market['B'],
                         'matched':len(chosen),'unmatched':market['n']-len(chosen),
                         'feasible':validate(market,chosen),'objective':actual,
                         'oracle_objective':oracle['objective'],
                         'quality_pct':100*actual/max(oracle['objective'],1e-12),
                         'prediction_mae':float(np.mean(np.abs(predicted-market['utility']))),
                         'inference_seconds':infer,'solver_seconds':solver,'timestamp_utc':utc_now()})
    write_csv(output/'robustness_per_seed.csv',rows)
    summary = []
    for shift in (0,10,20,30,40):
        group = [r for r in rows if r['shift_pct']==shift]
        entry = {'shift_pct':shift,'completed_seeds':len(group)}
        for key in ('quality_pct','objective','matched','capacity','prediction_mae','inference_seconds','solver_seconds'):
            entry[key+'_mean'],entry[key+'_sd']=stats([r[key] for r in group])
        summary.append(entry)
    write_csv(output/'robustness_summary.csv',summary)
    save_metadata(output/'run_details.json',{'created_utc':utc_now(),'config':config,'source':source_info,
                 'shift_definition':'job-skill scaling and capacity intensity change',
                 'test_market_separate_from_training_markets':True})
    print(f'Robustness records: {len(rows)}')


if __name__ == '__main__':
    main()
