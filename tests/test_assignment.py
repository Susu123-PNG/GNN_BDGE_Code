"""Check matching constraints, capacity accounting, and LP relaxation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from gnn_bdge.data import procedural_profiles, make_market
from gnn_bdge.solvers import solve_highs, validate


def test_market_constraints():
    profile, _ = procedural_profiles(7)
    market = make_market(profile, 40, 11)
    assert market['m'] == 8
    assert market['total_capacity'] <= 40
    assert market['B'] == min(40, market['total_capacity'])
    mip = solve_highs(market, market['utility'], 'mip', time_limit=10)
    lp = solve_highs(market, market['utility'], 'lp', time_limit=10)
    assert mip['feasible'] and lp['feasible']
    assert validate(market, mip['selected'])
    assert np.isclose(mip['objective'], lp['upper_bound'], atol=1e-6)


if __name__ == '__main__':
    test_market_constraints()
    print('All assignment checks passed.')
