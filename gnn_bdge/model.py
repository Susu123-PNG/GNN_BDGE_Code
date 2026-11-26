"""Relation-specific two-layer attention encoder and pairwise utility heads."""
import math
import torch
from torch import nn
import torch.nn.functional as F


def graph_inputs(market, device='cpu'):
    workers = torch.as_tensor(market['workers'], dtype=torch.float32, device=device)
    jobs = torch.as_tensor(market['jobs'], dtype=torch.float32, device=device)
    k, n, m = market['k'], market['n'], market['m']
    features = torch.cat((torch.cat((workers, torch.zeros(n, 1, device=device)), -1),
                          torch.cat((jobs, torch.as_tensor(market['capacities'], dtype=torch.float32, device=device)[:, None] / 5), -1),
                          torch.cat((torch.eye(k, device=device), torch.zeros(k, 1, device=device)), -1)), 0)
    sources, destinations, relation_ids = [], [], []
    for prefix, array, relation in ((0, workers, 0), (n, jobs, 1)):
        src, skill = torch.where(array > 0.40)
        left = src + prefix
        right = skill + n + m
        sources.extend((left, right))
        destinations.extend((right, left))
        relation_ids.extend((torch.full_like(left, relation), torch.full_like(right, relation + 2)))
    return features, torch.cat(sources), torch.cat(destinations), torch.cat(relation_ids)


class RelationAttention(nn.Module):
    def __init__(self, dim, heads=4, relations=4):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.width = dim // heads
        self.self_map = nn.Linear(dim, dim)
        self.query = nn.Linear(dim, dim, bias=False)
        self.keys = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(relations))
        self.values = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(relations))
        self.norm = nn.LayerNorm(dim)

    def forward(self, nodes, source, dest, relations):
        count = nodes.size(0)
        q = self.query(nodes).reshape(count, self.heads, self.width)
        accumulated = torch.zeros_like(q)
        for r, (key_fn, value_fn) in enumerate(zip(self.keys, self.values)):
            mask = relations == r
            if not torch.any(mask):
                continue
            u, v = source[mask], dest[mask]
            keys = key_fn(nodes[u]).reshape(-1, self.heads, self.width)
            values = value_fn(nodes[u]).reshape(-1, self.heads, self.width)
            scores = (q[v] * keys).sum(-1) / math.sqrt(self.width)
            # A separate softmax is computed for each receiving node and relation.
            maximum = torch.full((count, self.heads), -1e9, device=nodes.device, dtype=nodes.dtype)
            maximum.scatter_reduce_(0, v[:, None].expand(-1, self.heads), scores, reduce='amax', include_self=True)
            weights = torch.exp(scores - maximum[v])
            sums = torch.zeros((count, self.heads), device=nodes.device, dtype=nodes.dtype)
            sums.index_add_(0, v, weights)
            weights = weights / sums[v].clamp_min(1e-8)
            accumulated.index_add_(0, v, weights[:, :, None] * values)
        return F.elu(self.norm(self.self_map(nodes) + accumulated.reshape(count, -1)))


class HeterogeneousGAT(nn.Module):
    def __init__(self, k=30, hidden=64, heads=4):
        super().__init__()
        self.input_layer = nn.Linear(k + 1, hidden)
        self.layer1 = RelationAttention(hidden, heads)
        self.layer2 = RelationAttention(hidden, heads)
        self.bilinear = nn.Parameter(torch.randn(hidden, hidden) * 0.005)
        self.concat_weight = nn.Parameter(torch.randn(hidden * 2) * 0.005)
        self.product_weight = nn.Parameter(torch.randn(hidden) * 0.005)
        self.bias = nn.Parameter(torch.zeros(()))

    def encode(self, market):
        nodes, source, dest, relation_ids = graph_inputs(market)
        hidden = F.elu(self.input_layer(nodes))
        hidden = self.layer1(hidden, source, dest, relation_ids)
        hidden = self.layer2(hidden, source, dest, relation_ids)
        return hidden[:market['n']], hidden[market['n']:market['n']+market['m']]

    def score(self, embeddings, worker_idx, job_idx, batch_size=8192):
        seekers, jobs = embeddings
        output = []
        for start in range(0, len(worker_idx), batch_size):
            wi = torch.as_tensor(worker_idx[start:start+batch_size], dtype=torch.long)
            ji = torch.as_tensor(job_idx[start:start+batch_size], dtype=torch.long)
            u, v = seekers[wi], jobs[ji]
            interaction = ((u @ self.bilinear) * v).sum(-1)
            additive = torch.cat((u, v), -1) @ self.concat_weight
            products = (u * v) @ self.product_weight
            output.append(interaction + additive + products + self.bias)
        return torch.cat(output) if output else torch.empty(0)

    def forward(self, market, worker_idx=None, job_idx=None):
        if worker_idx is None:
            worker_idx = market['worker_index']
        if job_idx is None:
            job_idx = market['job_index']
        return self.score(self.encode(market), worker_idx, job_idx)


class PairMLP(nn.Module):
    def __init__(self, k=30, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2*k, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                                 nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, market, worker_idx, job_idx, batch_size=8192):
        u = torch.as_tensor(market['workers'], dtype=torch.float32)
        v = torch.as_tensor(market['jobs'], dtype=torch.float32)
        out = []
        for start in range(0, len(worker_idx), batch_size):
            i = torch.as_tensor(worker_idx[start:start+batch_size], dtype=torch.long)
            j = torch.as_tensor(job_idx[start:start+batch_size], dtype=torch.long)
            out.append(self.net(torch.cat((u[i], v[j]), -1)).reshape(-1))
        return torch.cat(out) if out else torch.empty(0)
