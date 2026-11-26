"""Create plots from seed-level metrics without replacing the underlying records."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as file:
        return list(csv.DictReader(file))


def main():
    base = Path('output')
    figures = base / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    primary = read_csv(base/'executed_demo'/'summary_by_method.csv')
    for method in ('GNN_BDGE','Greedy','Gale_Shapley','MLP_KM','Transport_Projection'):
        rows = sorted((r for r in primary if r['method']==method), key=lambda r: int(r['N']))
        plt.errorbar([int(r['N']) for r in rows], [float(r['quality_pct_mean']) for r in rows],
                     yerr=[float(r['quality_pct_sd']) for r in rows], capsize=3, label=method)
    plt.xlabel('Number of workers')
    plt.ylabel('Objective relative to same-instance optimum (%)')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figures/'assignment_quality.png', dpi=160)
    plt.close()
    graph = read_csv(base/'executed_demo'/'summary_by_method.csv')
    independent = read_csv(base/'non_graph_control'/'summary_by_method.csv')
    labels = ['GNN_BDGE','Greedy','MLP_KM','Transport_Projection']
    structural = [float(next(r for r in graph if r['N']=='200' and r['method']==method)['quality_pct_mean']) for method in labels]
    unrelated = [float(next(r for r in independent if r['N']=='200' and r['method']==method)['quality_pct_mean']) for method in labels]
    x = range(len(labels))
    plt.bar([i-.2 for i in x], structural, .4, label='Skill-related reward')
    plt.bar([i+.2 for i in x], unrelated, .4, label='Independent reward')
    plt.xticks(list(x), labels, rotation=16)
    plt.ylabel('Objective relative to same-instance optimum (%)')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figures/'reward_structure_control.png', dpi=160)
    plt.close()
    shifts = read_csv(base/'robustness'/'robustness_summary.csv')
    plt.errorbar([int(r['shift_pct']) for r in shifts], [float(r['quality_pct_mean']) for r in shifts],
                 yerr=[float(r['quality_pct_sd']) for r in shifts], capsize=3)
    plt.xlabel('Applied distribution shift (%)')
    plt.ylabel('Relative assignment objective (%)')
    plt.tight_layout()
    plt.savefig(figures/'shift_sensitivity.png', dpi=160)
    plt.close()
    sensitivities = [r for r in read_csv(base/'spsa_sensitivity'/'spsa_parameter_summary.csv') if r['delta']=='0.0001']
    plt.plot([int(r['P']) for r in sensitivities], [float(r['gradient_norm_mean']) for r in sensitivities], marker='o')
    plt.xlabel('SPSA direction count')
    plt.ylabel('Average gradient norm')
    plt.tight_layout()
    plt.savefig(figures/'spsa_gradient_norm.png', dpi=160)
    plt.close()
    print(f'Saved four figures to {figures}')


if __name__ == '__main__':
    main()
