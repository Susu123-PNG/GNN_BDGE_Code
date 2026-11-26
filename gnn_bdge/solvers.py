"""Feasible assignment routines with independent solver timing and status records."""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp, linear_sum_assignment
from scipy.sparse import coo_matrix


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def logger(path, event, **values):
    if not path:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {'timestamp_utc': utc_now(), 'event': event, **values}
    with path.open('a', encoding='utf-8') as file:
        file.write(json.dumps(record, ensure_ascii=True, default=str) + '\n')


def matrix_for(market):
    worker = market['worker_index']
    job = market['job_index']
    count = len(worker)
    cols = np.arange(count)
    row_ids = np.concatenate((worker, market['n'] + job))
    col_ids = np.concatenate((cols, cols))
    mat = coo_matrix((np.ones(2*count), (row_ids, col_ids)),
                     shape=(market['n']+market['m'], count)).tocsr()
    b = np.concatenate((np.ones(market['n']), market['capacities']))
    return mat, b


def validate(market, selected):
    worker = np.bincount(market['worker_index'][selected], minlength=market['n'])
    job = np.bincount(market['job_index'][selected], minlength=market['m'])
    return bool((worker <= 1).all() and (job <= market['capacities']).all())


def solve_highs(market, values, mode='mip', time_limit=60.0, mip_gap=1e-6,
                node_limit=None, log_file=None):
    mat, upper = matrix_for(market)
    scores = np.asarray(values, dtype=float)
    t = time.perf_counter()
    logger(log_file, 'solver_started', backend='HiGHS (SciPy)', scipy_mode=mode,
           candidate_edges=len(scores), time_limit_seconds=time_limit, mip_rel_gap=mip_gap, node_limit=node_limit)
    if mode == 'lp':
        result = linprog(-scores, A_ub=mat, b_ub=upper, bounds=(0, 1), method='highs',
                         options={'time_limit': time_limit, 'dual_feasibility_tolerance': 1e-7})
        x = result.x
        dual_bound = -result.fun if result.fun is not None else None
        mip_gap_result = None
    elif mode == 'mip':
        opts = {'time_limit': time_limit, 'mip_rel_gap': mip_gap}
        if node_limit is not None:
            opts['node_limit'] = int(node_limit)
        result = milp(-scores, integrality=np.ones(len(scores), dtype=int), bounds=Bounds(0, 1),
                      constraints=LinearConstraint(mat, -np.inf, upper), options=opts)
        x = result.x
        dual_bound = -float(result.mip_dual_bound) if getattr(result, 'mip_dual_bound', None) is not None else None
        mip_gap_result = getattr(result, 'mip_gap', None)
    else:
        raise ValueError('mode must be lp or mip')
    elapsed = time.perf_counter() - t
    successful = x is not None and np.all(np.isfinite(x))
    chosen = np.flatnonzero(x >= 1 - 1e-5) if successful else np.empty(0, dtype=int)
    objective = float(scores[chosen].sum()) if successful and validate(market, chosen) else None
    payload = {'selected': chosen, 'objective': objective, 'wall_seconds': elapsed,
               'status': int(result.status), 'status_message': str(result.message),
               'feasible': objective is not None, 'upper_bound': dual_bound,
               'gap': None if mip_gap_result is None else float(mip_gap_result),
               'fractional_variables': int(np.sum((x > 1e-5) & (x < 1 - 1e-5))) if successful else None}
    logger(log_file, 'solver_finished', backend='HiGHS (SciPy)', mode=mode,
           **{k: v for k, v in payload.items() if k != 'selected'})
    return payload


def solve_gurobi(market, values, time_limit=3600, mip_gap=1e-6, threads=None, log_file=None):
    # The commercial backend is optional; an absent installation is never reported as a solve.
    import gurobipy as gp
    from gurobipy import GRB
    path = str(log_file) if log_file else ''
    model = gp.Model('assignment')
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = mip_gap
    if threads is not None:
        model.Params.Threads = int(threads)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        model.Params.LogFile = path
    x = model.addVars(len(values), vtype=GRB.BINARY, name='x')
    for i in range(market['n']):
        idx = np.flatnonzero(market['worker_index'] == i)
        if len(idx):
            model.addConstr(gp.quicksum(x[int(t)] for t in idx) <= 1)
    for j in range(market['m']):
        idx = np.flatnonzero(market['job_index'] == j)
        if len(idx):
            model.addConstr(gp.quicksum(x[int(t)] for t in idx) <= int(market['capacities'][j]))
    model.setObjective(gp.quicksum(float(values[t])*x[t] for t in range(len(values))), GRB.MAXIMIZE)
    t0 = time.perf_counter()
    model.optimize()
    elapsed = time.perf_counter() - t0
    selected = np.array([t for t in range(len(values)) if model.SolCount and x[t].X > 0.5], dtype=int)
    feasible = bool(model.SolCount and validate(market, selected))
    output = {'selected': selected, 'objective': float(np.asarray(values)[selected].sum()) if feasible else None,
              'wall_seconds': elapsed, 'status': int(model.Status), 'feasible': feasible,
              'upper_bound': float(model.ObjBound) if model.SolCount else None,
              'gap': float(model.MIPGap) if model.SolCount else None}
    model.dispose()
    return output


def greedy(market, values):
    order = np.argsort(-np.asarray(values), kind='stable')
    assigned = np.zeros(market['n'], dtype=bool)
    remaining = market['capacities'].copy()
    chosen = []
    for idx in order:
        w, j = market['worker_index'][idx], market['job_index'][idx]
        if values[idx] > 0 and not assigned[w] and remaining[j] > 0:
            chosen.append(idx)
            assigned[w] = True
            remaining[j] -= 1
    return np.asarray(chosen, dtype=int)


def lp_round(market, values, time_limit=40, log_file=None):
    result = solve_highs(market, values, 'lp', time_limit=time_limit, log_file=log_file)
    if not result['feasible']:
        # A failed LP is not treated as a successful optimization run.
        return {'selected': greedy(market, values), 'lp_status': result['status'], 'lp_feasible': False,
                'lp_upper_bound': result['upper_bound']}
    return {'selected': result['selected'], 'lp_status': result['status'], 'lp_feasible': True,
            'lp_upper_bound': result['upper_bound'], 'lp_fractional': result['fractional_variables']}


def deferred_acceptance(market, values):
    n, m = market['n'], market['m']
    workers = [[] for _ in range(n)]
    for t, (i, j) in enumerate(zip(market['worker_index'], market['job_index'])):
        workers[int(i)].append((float(values[t]), int(j), t))
    workers = [sorted(items, reverse=True) for items in workers]
    waiting = list(range(n))
    next_choice = np.zeros(n, dtype=int)
    accepted = [[] for _ in range(m)]
    while waiting:
        i = waiting.pop()
        if next_choice[i] >= len(workers[i]):
            continue
        val, j, t = workers[i][next_choice[i]]
        next_choice[i] += 1
        if val <= 0:
            continue
        accepted[j].append((val, i, t))
        accepted[j].sort(reverse=True)
        if len(accepted[j]) > market['capacities'][j]:
            _, expelled, _ = accepted[j].pop()
            waiting.append(expelled)
    return np.asarray([t for row in accepted for _, _, t in row], dtype=int)


def expanded_hungarian(market, values):
    # Empty columns permit an unmatched worker without inventing job capacity.
    n = market['n']
    slots = np.repeat(np.arange(market['m']), market['capacities'])
    pair_map = {(int(i), int(j)): idx for idx, (i, j) in enumerate(zip(market['worker_index'], market['job_index']))}
    cost = np.zeros((n, len(slots) + n), dtype=np.float64)
    cost[:, :len(slots)] = 1e4
    for col, j in enumerate(slots):
        rows = market['worker_index'][market['job_index'] == j]
        for i in rows:
            cost[i, col] = -max(float(values[pair_map[(int(i), int(j))]]), 0.0)
    row, col = linear_sum_assignment(cost)
    selected = [pair_map[(int(i), int(slots[c]))] for i, c in zip(row, col)
                if c < len(slots) and cost[i, c] < 0 and (int(i), int(slots[c])) in pair_map]
    return np.asarray(selected, dtype=int)


def transport_projection(market, values, temperature=0.10, iterations=60):
    # Entropic soft assignments provide priorities; capacity is enforced during projection.
    n, m = market['n'], market['m']
    dense = np.full((n, m + 1), -50., dtype=float)
    dense[:, m] = 0.0
    dense[market['worker_index'], market['job_index']] = values / temperature
    dense -= dense.max(axis=1, keepdims=True)
    matrix = np.exp(dense)
    for _ in range(iterations):
        matrix /= np.maximum(matrix.sum(axis=1, keepdims=True), 1e-12)
        counts = matrix[:, :m].sum(axis=0)
        scale = np.minimum(1.0, market['capacities'] / np.maximum(counts, 1e-12))
        matrix[:, :m] *= scale[None, :]
    priority = matrix[market['worker_index'], market['job_index']] * np.maximum(values, 0)
    return greedy(market, priority)
