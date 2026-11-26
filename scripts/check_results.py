"""Run basic quality checks on saved experiments."""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='output/standard')
    args = parser.parse_args()
    path = Path(args.results)
    with (path / 'raw_results_per_market.csv').open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    assert rows, 'No experiment records'
    for row in rows:
        assert row['feasible'] == 'True', row
        assert int(row['matched']) <= int(row['B']), row
        assert int(row['unmatched']) + int(row['matched']) == int(row['worker_count']), row
        computed = 100 * float(row['objective']) / float(row['reference_optimum'])
        assert math.isclose(computed, float(row['quality_pct']), rel_tol=1e-8), row
    print(f'Validated {len(rows)} experiment records.')


if __name__ == '__main__':
    main()
