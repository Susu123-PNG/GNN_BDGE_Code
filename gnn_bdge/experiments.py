"""Seed-level experiments, solver comparisons, and metric aggregation."""
import csv
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import torch

from .data import check_esco, load_onet_profiles, make_market, procedural_profiles, save_market, save_metadata
from .solvers import (deferred_acceptance, expanded_hungarian, greedy, logger,
                      lp_round, solve_gurobi, solve_highs, transport_projection, utc_now, validate)
from .training import fit_models


def write_csv(file, rows, names=None):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    if not rows and names is None:
        return
    if names is None:
        names = list(dict.fromkeys(k for row in rows for k in row))
    with file.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def save_assignments(file, market, indexes, score):
    indexes = np.asarray(indexes, dtype=int)
    rows = [{'worker': int(market['worker_index'][i]), 'job': int(market['job_index'][i]),
             'true_utility': float(market['utility'][i]), 'predicted_score': float(score[i])}
            for i in indexes]
    write_csv(file, rows, ['worker', 'job', 'true_utility', 'predicted_score'])


def stats(values):
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return (float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr) > 1 else float('nan')) if len(arr) else (float('nan'), float('nan'))


def run(config, result_dir, source=None, esco_dir=None, source_mode='onet31', use_gurobi=False):
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    sizes = [int(n) for n in config['sizes']]
    seeds = [int(n) for n in config['seeds']]
    if source_mode == 'onet31':
        if not source:
            raise ValueError('O*NET 31.0 archive or extracted CSV directory is required for onet31 mode.')
        profiles, provenance = load_onet_profiles(source, int(config['skill_dimensions']))
    elif source_mode == 'procedural':
        profiles, provenance = procedural_profiles(int(config['profile_seed']), int(config['skill_dimensions']))
    else:
        raise ValueError('source_mode must be onet31 or procedural')
    esco_check = check_esco(esco_dir)
    actual_hardware = {'machine': platform.machine(), 'processor': platform.processor(), 'cpu_count': os.cpu_count(),
                       'torch_cuda_available': torch.cuda.is_available(),
                       'cuda_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                       'python': sys.version.split()[0], 'numpy': np.__version__, 'torch': torch.__version__,
                       'scipy': scipy.__version__, 'platform': platform.platform()}
    run_details = {'started_at_utc': utc_now(), 'source': provenance, 'esco': esco_check, 'utility_mode': config.get('utility_mode', 'structural'),
                   'hardware': actual_hardware, 'actual_configuration': config,
                   'model_class': 'capacitated bipartite assignment',
                   'capacity_rule': 'integer capacity from clipped Poisson, one unit per assigned worker',
                   'unmatched_workers_allowed': True,
                   'qualification_rule': f"top {config['required_skills']} job skills, worker >= {config['qualification_ratio']} * job requirement",
                   'capacity_mean_assumption': config['capacity_mean'],
                   'external_expert_skill_alignment_verified': False,
                   'data_7_1_2_split': 'independent market instances',
                   'metric_denominator': 'same-instance objective of HiGHS integer optimization',
                   'gurobi_requested': bool(use_gurobi)}
    save_metadata(result_dir / 'run_details.json', run_details)
    rows, diagnostics, training_rows, sensitivity_rows = [], [], [], []
    for size in sizes:
        for seed in seeds:
            local_dir = result_dir / 'seeds' / f'N{size}_seed{seed}'
            local_dir.mkdir(parents=True, exist_ok=True)
            market_kwargs = dict(n=size, k=config['skill_dimensions'], capacity_mean=config['capacity_mean'],
                                 tau=config['candidate_threshold'], noise_std=config['worker_noise'],
                                 required_skills=config['required_skills'], minimum_ratio=config['qualification_ratio'], utility_mode=config.get('utility_mode', 'structural'))
            markets = [make_market(profiles, seed=seed * 1000 + i, **market_kwargs) for i in range(10)]
            train, val, test = markets[:7], markets[7], markets[8:]
            save_market(test[0], local_dir / 'market_instance.npz')
            write_csv(local_dir / 'market_edges.csv',
                      [{'worker': int(i), 'job': int(j), 'true_utility': float(v)}
                       for i,j,v in zip(test[0]['worker_index'], test[0]['job_index'], test[0]['utility'])],
                      ['worker', 'job', 'true_utility'])
            write_csv(local_dir / 'job_capacities.csv',
                      [{'job': j, 'capacity': int(v)} for j,v in enumerate(test[0]['capacities'])])
            model, mlp, train_log = fit_models(train, val, config, seed)
            torch.save(model.state_dict(), local_dir / 'gnn_weights.pt')
            torch.save(mlp.state_dict(), local_dir / 'mlp_weights.pt')
            write_csv(local_dir / 'train_losses.csv', train_log['loss_trace'])
            write_csv(local_dir / 'spsa_trace.csv', train_log['gradient_trace'],
                      ['step','gradient_norm','difference_mean','decision_loss_after_update'])
            training_rows.append({'N': size, 'seed': seed, 'source_mode': source_mode,
                                  'training_seconds': train_log['train_wall_seconds'],
                                  'train_epochs': train_log['actual_training_epochs'],
                                  'spsa_steps': train_log['actual_spsa_steps'],
                                  'spsa_directions': train_log['actual_directions']})
            all_seed_rows = []
            for test_num, market in enumerate(test):
                trace = local_dir / f'HiGHS_test{test_num}.jsonl'
                if trace.exists():
                    trace.unlink()
                oracle = solve_highs(market, market['utility'], 'mip',
                                     time_limit=config['solver_time_limit'], mip_gap=config['mip_gap'], log_file=trace)
                lp = solve_highs(market, market['utility'], 'lp',
                                 time_limit=config['solver_time_limit'], log_file=trace)
                if not oracle['feasible']:
                    logger(trace, 'instance_skipped', reason='reference integer model has no feasible solution')
                    continue
                optimum = oracle['objective']
                if optimum <= 0:
                    logger(trace, 'instance_skipped', reason='reference objective is non-positive')
                    continue
                t0 = time.perf_counter()
                model.eval()
                with torch.no_grad():
                    gnn_scores = model(market).cpu().numpy().astype(np.float64)
                gnn_forward = time.perf_counter() - t0
                t0 = time.perf_counter()
                mlp.eval()
                with torch.no_grad():
                    mlp_scores = mlp(market, market['worker_index'], market['job_index']).cpu().numpy().astype(np.float64)
                mlp_forward = time.perf_counter() - t0
                write_csv(local_dir / f'candidate_scores_test{test_num}.csv',
                          [{'worker': int(i), 'job': int(j), 'true_utility': float(y),
                            'gnn_score': float(a), 'mlp_score': float(b)}
                           for i, j, y, a, b in zip(market['worker_index'], market['job_index'],
                                                     market['utility'], gnn_scores, mlp_scores)],
                          ['worker', 'job', 'true_utility', 'gnn_score', 'mlp_score'])
                def method_record(label, chosen, predictions, inference_seconds, solver_seconds, solve_status='ok',
                                  score_upper=None, report=True):
                    chosen = np.asarray(chosen, dtype=int)
                    valid = validate(market, chosen)
                    achieved = float(market['utility'][chosen].sum()) if valid else float('nan')
                    predicted_obj = float(predictions[chosen].sum()) if valid else float('nan')
                    quality = 100.0 * achieved / optimum if valid else float('nan')
                    record = {'N': size, 'seed': seed, 'test_instance': test_num,
                              'source_mode': source_mode, 'method': label,
                              'worker_count': market['n'], 'job_count': market['m'],
                              'candidate_edges': len(market['utility']),
                              'total_capacity': market['total_capacity'], 'B': market['B'],
                              'matched': len(chosen), 'unmatched': market['n']-len(chosen),
                              'assignment_rate_pct': 100 * len(chosen)/market['n'],
                              'capacity_utilization_pct': 100 * len(chosen)/market['total_capacity'],
                              'feasible': valid, 'objective': achieved, 'predicted_objective': predicted_obj,
                              'reference_optimum': optimum, 'lp_upper_bound': lp['upper_bound'],
                              'quality_pct': quality, 'relative_gap_pct': 100. - quality,
                              'reference_gap_pct': 100 * max(0, (lp['upper_bound'] - achieved) / lp['upper_bound']) if lp['upper_bound'] else float('nan'),
                              'prediction_mae': float(np.mean(np.abs(predictions - market['utility']))),
                              'prediction_max_abs': float(np.max(np.abs(predictions - market['utility']))),
                              'inference_seconds': inference_seconds, 'solver_seconds': solver_seconds,
                              'end_to_end_seconds': inference_seconds + solver_seconds,
                              'solver_status': solve_status, 'timestamp_utc': utc_now()}
                    all_seed_rows.append(record)
                    rows.append(record)
                    if report:
                        save_assignments(local_dir / f'{label}_test{test_num}_assignment.csv', market, chosen, predictions)
                    if score_upper is not None:
                        empirical_epsilon = max(0.0, float(score_upper)-predicted_obj)
                        empirical_delta = float(np.max(np.abs(predictions - market['utility'])))
                        diagnostics.append({'N': size, 'seed': seed, 'test_instance': test_num,
                                            'method': label, 'B': market['B'], 'empirical_delta': empirical_delta,
                                            'lower_optimization_error': empirical_epsilon,
                                            'empirical_regret': optimum-achieved,
                                            'error_bound_rhs': empirical_epsilon + 2*market['B']*empirical_delta,
                                            'inequality_holds': optimum-achieved <= empirical_epsilon + 2*market['B']*empirical_delta + 1e-5})
                method_record('HiGHS_MIP_Oracle', oracle['selected'], market['utility'], 0., oracle['wall_seconds'], oracle['status_message'])
                method_record('HiGHS_LP', lp['selected'], market['utility'], 0., lp['wall_seconds'], lp['status_message'])
                t0 = time.perf_counter()
                pred_lp = lp_round(market, gnn_scores, time_limit=config['solver_time_limit'], log_file=trace)
                lower_time = time.perf_counter() - t0
                method_record('GNN_BDGE', pred_lp['selected'], gnn_scores, gnn_forward, lower_time,
                              'LP feasible' if pred_lp['lp_feasible'] else 'LP fallback', pred_lp['lp_upper_bound'])
                for label, procedure, scores, forward in (
                    ('Greedy', greedy, gnn_scores, gnn_forward),
                    ('Gale_Shapley', deferred_acceptance, gnn_scores, gnn_forward),
                    ('MLP_KM', expanded_hungarian, mlp_scores, mlp_forward),
                    ('Transport_Projection', transport_projection, gnn_scores, gnn_forward)):
                    t0 = time.perf_counter()
                    chosen = procedure(market, scores)
                    method_record(label, chosen, scores, forward, time.perf_counter()-t0)
                diagnostics.append({'N': size, 'seed': seed, 'test_instance': test_num,
                                    'method': 'LP_MIP_consistency', 'lp_upper_bound': lp['upper_bound'],
                                    'mip_objective': optimum,
                                    'absolute_difference': abs(lp['upper_bound'] - optimum) if lp['upper_bound'] else float('nan'),
                                    'lp_fractional_variables': lp['fractional_variables'],
                                    'capacity_less_than_workers': market['total_capacity'] < size})
                if use_gurobi:
                    try:
                        result = solve_gurobi(market, market['utility'],
                                              time_limit=config['gurobi_time_limit'], mip_gap=config['mip_gap'],
                                              log_file=local_dir / f'Gurobi_test{test_num}.log')
                        if result['feasible']:
                            method_record('Gurobi_MIP_Oracle', result['selected'], market['utility'],
                                          0., result['wall_seconds'], str(result['status']))
                    except Exception as exc:
                        # An environment or licensing problem must be visible to the caller.
                        save_metadata(local_dir / f'Gurobi_test{test_num}_not_run.json',
                                      {'timestamp_utc': utc_now(), 'status': 'not_run',
                                       'reason': type(exc).__name__ + ': ' + str(exc)})
            write_csv(local_dir / 'method_results.csv', all_seed_rows)
            print(f'completed N={size} seed={seed} valid_test_markets={len(test)}', flush=True)
    write_csv(result_dir / 'raw_results_per_market.csv', rows)
    write_csv(result_dir / 'training_by_seed.csv', training_rows)
    write_csv(result_dir / 'diagnostics.csv', diagnostics)
    # Means are taken within seed before across-seed standard deviations.
    per_seed = []
    for size in sizes:
        for seed in seeds:
            methods = sorted({r['method'] for r in rows if r['N']==size and r['seed']==seed})
            for method in methods:
                group = [r for r in rows if r['N']==size and r['seed']==seed and r['method']==method]
                result = {'N': size, 'seed': seed, 'method': method, 'test_markets': len(group)}
                for key in ('quality_pct','relative_gap_pct','objective','matched','assignment_rate_pct',
                            'capacity_utilization_pct','prediction_mae','prediction_max_abs',
                            'solver_seconds','inference_seconds','end_to_end_seconds'):
                    result[key] = stats([r[key] for r in group])[0]
                per_seed.append(result)
    write_csv(result_dir / 'raw_results_per_seed.csv', per_seed)
    summary = []
    for size in sizes:
        methods = sorted({r['method'] for r in per_seed if r['N']==size})
        for method in methods:
            group = [r for r in per_seed if r['N']==size and r['method']==method]
            row = {'N': size, 'method': method, 'completed_seeds': len(group), 'source_mode': source_mode}
            for key in ('quality_pct','relative_gap_pct','objective','matched','assignment_rate_pct',
                        'capacity_utilization_pct','prediction_mae','prediction_max_abs',
                        'solver_seconds','inference_seconds','end_to_end_seconds'):
                row[key+'_mean'], row[key+'_sd'] = stats([r[key] for r in group])
            summary.append(row)
    write_csv(result_dir / 'summary_by_method.csv', summary)
    run_details.update({'finished_at_utc': utc_now(), 'completed_seeds': len(seeds)*len(sizes),
                        'completed_test_records': len(rows), 'result_status': 'executed'})
    save_metadata(result_dir / 'run_details.json', run_details)
    return summary
