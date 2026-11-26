"""Supervised utility learning followed by decision-feedback parameter updates."""
import time
import numpy as np
import torch
from torch import nn

from .model import HeterogeneousGAT, PairMLP
from .solvers import solve_highs


def sample_pairs(market, limit, seed):
    count = len(market['utility'])
    if count <= limit:
        return np.arange(count)
    return np.random.default_rng(seed).choice(count, size=limit, replace=False)


def _decision_loss(model, market, reference_value, time_limit=15):
    model.eval()
    with torch.no_grad():
        score = model(market).cpu().numpy().astype(np.float64)
    solution = solve_highs(market, score, 'lp', time_limit=time_limit)
    selection = solution['selected']
    true_value = float(market['utility'][selection].sum())
    return max(0.0, 1 - true_value / max(reference_value, 1e-12))


def fit_models(train_markets, val_market, config, seed, log_path=None):
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, int(config.get('torch_threads', 2))))
    k = train_markets[0]['k']
    model = HeterogeneousGAT(k, config['hidden_dim'], config['attention_heads'])
    mlp = PairMLP(k, config['hidden_dim'])
    opt = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    opt_mlp = torch.optim.Adam(mlp.parameters(), lr=config['learning_rate'])
    epochs = int(config['train_epochs'])
    steps = int(config['spsa_steps'])
    loss_trace = []
    t_start = time.perf_counter()
    for epoch in range(epochs):
        market = train_markets[epoch % len(train_markets)]
        indexes = sample_pairs(market, int(config['training_pairs_per_epoch']), seed + epoch * 17)
        wi = market['worker_index'][indexes]
        ji = market['job_index'][indexes]
        target = torch.as_tensor(market['utility'][indexes], dtype=torch.float32)
        model.train()
        opt.zero_grad()
        batch_size = int(config['batch_size'])
        embeddings = model.encode(market)
        mse_total = torch.zeros((), dtype=torch.float32)
        for start in range(0, len(indexes), batch_size):
            stop = start + batch_size
            predicted = model.score(embeddings, wi[start:stop], ji[start:stop])
            mse_total = mse_total + ((predicted - target[start:stop]) ** 2).sum()
        mse = mse_total / len(indexes)
        ridge = sum((p * p).sum() for p in model.parameters())
        full_loss = mse + config['weight_decay'] * ridge
        full_loss.backward()
        opt.step()
        mlp.train()
        opt_mlp.zero_grad()
        mlp_total = torch.zeros((), dtype=torch.float32)
        for start in range(0, len(indexes), batch_size):
            stop = start + batch_size
            predicted = mlp(market, wi[start:stop], ji[start:stop])
            mlp_total = mlp_total + ((predicted - target[start:stop]) ** 2).sum()
        mlp_loss = mlp_total / len(indexes)
        mlp_loss.backward()
        opt_mlp.step()
        loss_trace.append({'epoch': epoch + 1, 'gnn_mse': float(mse.detach()),
                           'mlp_mse': float(mlp_loss.detach())})
    gradient_trace = []
    if steps:
        oracle = solve_highs(val_market, val_market['utility'], 'lp', time_limit=config['solver_time_limit'])
        true_opt = oracle['objective']
        if true_opt is None:
            raise RuntimeError('Validation reference solve did not return a feasible allocation.')
        rng = np.random.default_rng(seed + 99173)
        params = [p for p in model.parameters() if p.requires_grad]
        directions = int(config['spsa_directions'])
        delta = float(config['spsa_delta'])
        for step in range(steps):
            gradient = [torch.zeros_like(p) for p in params]
            differences = []
            for _ in range(directions):
                noise = [torch.as_tensor(rng.choice([-1., 1.], size=p.shape), dtype=p.dtype) for p in params]
                with torch.no_grad():
                    for p, direction in zip(params, noise):
                        p.add_(delta * direction)
                loss_plus = _decision_loss(model, val_market, true_opt, config['solver_time_limit'])
                with torch.no_grad():
                    for p, direction in zip(params, noise):
                        p.add_(-2 * delta * direction)
                loss_minus = _decision_loss(model, val_market, true_opt, config['solver_time_limit'])
                with torch.no_grad():
                    for p, direction in zip(params, noise):
                        p.add_(delta * direction)
                coefficient = (loss_plus - loss_minus) / (2 * delta)
                differences.append(coefficient)
                for grad, direction in zip(gradient, noise):
                    grad.add_(coefficient * direction / directions)
            norm = float(torch.sqrt(sum((g*g).sum() for g in gradient)))
            with torch.no_grad():
                for p, g in zip(params, gradient):
                    p.add_(-config['spsa_learning_rate'] * g)
            gradient_trace.append({'step': step + 1, 'gradient_norm': norm,
                                   'difference_mean': float(np.mean(differences)),
                                   'decision_loss_after_update': _decision_loss(model, val_market, true_opt,
                                                                              config['solver_time_limit'])})
    elapsed = time.perf_counter() - t_start
    return model, mlp, {'train_wall_seconds': elapsed, 'loss_trace': loss_trace,
                        'gradient_trace': gradient_trace,
                        'actual_training_epochs': epochs, 'actual_spsa_steps': steps,
                        'actual_directions': int(config['spsa_directions'])}
