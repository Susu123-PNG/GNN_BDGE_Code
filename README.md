# GNN-BDGE: Graph Neural Network with Bilevel Decomposition and Generalization Error Bounds

## 1. Project overview

GNN-BDGE combines heterogeneous graph representation learning, decision-feedback training, and capacity-constrained assignment optimization for worker-to-job matching. Worker attributes, job requirements, and skills form a typed graph. A graph attention encoder estimates candidate-pair utilities, while an optimization backend selects assignments under eligibility and capacity constraints. The project also provides reference optimization routines, matching baselines, parameter studies, and structured experiment outputs.

## 2. Data and market construction

### Data interfaces

- **O*NET 31.0:** The `onet31` input mode reads `essential_skills.csv` and `transferable_skills.csv` from the official CSV archive or an extracted directory; importance-scale (`IM`) values are combined into occupation-level skill profiles.
- **ESCO 1.2.1:** An optional ESCO directory supplies `occupations_en.csv`, `skills_en.csv`, and `occupationSkillRelations_en.csv` for file-level inventory checks.
- **Occupation-profile input modes:** `onet31` uses the supplied O*NET archive, and `procedural` constructs occupation profiles from seeded numerical distributions.
- **Market instances:** Workers are sampled from occupation profiles with Gaussian skill noise, while job skill requirements are drawn from occupation profiles.

### Market and qualification rules

Let `N` denote the number of workers, `M` the number of jobs, and `K` the number of skill dimensions. Each market uses `M = N / 5` (for the configured sizes), `K = 30`, and a discrete job capacity `c_j` drawn from a Poisson distribution with mean `3.0` and clipped to `[1, 5]`.

A worker–job candidate is retained if (i) the worker satisfies the required skills associated with that job and (ii) the cosine similarity of their skill vectors is at least `0.1`. The implementation selects the two highest-valued job skills as mandatory skills and requires the worker's corresponding values to be at least `0.45` times the job requirements. Worker skill vectors use Gaussian noise with standard deviation `0.12`. In the structural utility mode, candidate-pair utility equals cosine similarity between worker and job skill vectors.

The default experiment sizes are `N ∈ {200, 500, 1000, 2000}` with five random seeds. Markets are split by independent market instance, with seven training markets, one validation market, and two test markets per size and seed (`7:1:2`).

## 3. Assignment model

Let `E` be the eligible worker–job pairs and `x_ij ∈ {0,1}` indicate whether worker `i` is assigned to job `j`. Given pairwise scores `s_ij`, the assignment model is

```text
maximize    sum_(i,j) in E  s_ij * x_ij
subject to  sum_j x_ij <= 1                 for each worker i
            sum_i x_ij <= c_j               for each job j
            x_ij = 0                        for (i,j) outside E
            x_ij in {0,1}                   for (i,j) in E
```

Unassigned workers are allowed. The instance-specific upper bound on the number of assignments is `B = min(N, sum_j c_j)`; eligibility constraints may reduce the achievable count. Feasibility checks cover worker uniqueness, job capacities, and the permitted candidate set.

Reference utility and predicted optimization utility are evaluated separately: the optimization backend selects assignments using its supplied scores, and reported realized utility is calculated from the market's utility values.

## 4. Graph encoder and decision learning

### Heterogeneous graph

The graph has three node types: workers, jobs, and skills. Worker–skill and job–skill links are constructed from positive skill associations, with a feature threshold of `0.40` in the current graph builder; reverse relations are also included. The encoder uses relation-specific attention on these directed edge types.

### Utility scoring network

The encoder contains two graph attention layers with a hidden representation size of `64` and `4` attention heads. For the final worker representation `h_i` and job representation `h_j`, the pairwise scoring head is

```text
s_ij = h_i^T Q h_j + w^T [h_i || h_j] + v^T (h_i * h_j) + b
```

Here, `||` denotes concatenation and `*` denotes elementwise multiplication. Each relation uses its own key and value projections. The model applies ELU nonlinearities and layer normalization in attention layers.

### Training and decision feedback

Supervised utility estimation uses mean squared error with L2 parameter regularization. Adam updates the graph encoder and scoring parameters. A subsequent decision-feedback stage estimates directional derivatives through symmetric parameter perturbations (SPSA) and updates the parameters using the assignment decision loss. The decision loss is based on the relative difference between the reference assignment utility and the realized utility of the selected assignment.

## 5. Standard configuration

The values below are defined in `config/standard.json`.

| Setting | Value |
| --- | --- |
| Worker counts `N` | `200, 500, 1000, 2000` |
| Job count `M` | `0.2 × N` |
| Skill dimensions | `30` |
| Random seeds | `41, 73, 107, 149, 181` |
| Training / validation / test markets | `7 / 1 / 2` |
| Job capacity distribution | Poisson mean `3.0`, clipped to `1–5` |
| Worker skill-noise standard deviation | `0.12` |
| Candidate cosine threshold | `0.1` |
| Mandatory job skills | `2` |
| Mandatory skill ratio | `0.45` |
| Graph feature-link threshold | `0.40` (set in `gnn_bdge/model.py`) |
| Graph attention layers | `2` |
| Hidden dimension | `64` |
| Attention heads | `4` |
| Optimizer | Adam |
| Initial learning rate | `1e-3` |
| Weight decay | `1e-5` |
| Batch size | `32` |
| Training epochs | `200` |
| Training pairs sampled per epoch | `2048` |
| SPSA update steps | `200` |
| SPSA directions `P` | `10` |
| SPSA perturbation `delta` | `1e-4` |
| SPSA parameter update rate | `1e-6` |
| Solver time limit | `3600` seconds |
| Relative MIP gap tolerance | `1e-6` |

`config/quick.json` provides a separate compact configuration with the same graph dimensions and market-generation rules but shorter optimization budgets; the selected configuration is recorded in each run's metadata.

## 6. Optimization methods and measurements

The HiGHS backend is accessed through `scipy.optimize.linprog(method="highs")` for linear programming and `scipy.optimize.milp` for mixed-integer optimization. A Gurobi backend is available through `gurobipy` when installed and enabled. The graph-based assignment path computes GNN scores, solves the corresponding capacity-constrained linear program, and checks the selected allocation. Additional methods include greedy allocation, Gale–Shapley deferred acceptance, an MLP with expanded-slot Hungarian matching, and entropic transport priorities with capacity-aware projection.

The results use the following metric definitions:

| Metric | Definition |
| --- | --- |
| Realized objective | Sum of actual utilities on selected worker–job pairs |
| Matching quality (%) | `100 × realized objective / reference integer optimum` |
| Relative gap (%) | `100 − matching quality (%)` |
| Matched / unmatched | Assigned worker count / `N − assigned worker count` |
| Assignment rate (%) | `100 × matched / N` |
| Capacity utilization (%) | `100 × matched / total job capacity` |
| Feasibility | Worker uniqueness and job-capacity constraints are satisfied |
| Prediction MAE | Mean absolute error between predicted and actual candidate-pair utilities |
| Maximum prediction error | Largest absolute candidate-pair prediction error |
| Solver time | Time spent in the assignment algorithm or optimization backend |
| Inference time | Time spent producing model scores |
| End-to-end inference time | Inference time plus assignment/solver time, excluding offline training |

Per-seed records average the two held-out market results for each method. The final summary reports the mean and sample standard deviation (`ddof=1`) over seeds. Diagnostics also contain the LP objective, MIP objective, lower-level optimization error, empirical utility-prediction error, and the associated regret inequality check.

## 7. Installation and commands

Run the following commands from the project root.

```bash
python -m pip install -r requirements.txt
python scripts/check_environment.py
python scripts/download_sources.py --target onet31 --directory data/input
python scripts/download_sources.py --target esco_onet_crosswalk --directory data/input
```

Run the standard O*NET-based configuration:

```bash
python scripts/run_experiments.py \
  --config config/standard.json \
  --source-mode onet31 \
  --onet data/input/onet31.csv.zip \
  --results output/standard
```

The `--esco` option accepts a directory containing extracted ESCO CSV files. Use `--gurobi` to enable the optional Gurobi reference backend. A self-contained profile input is also selectable with `--source-mode procedural`.

Additional experiment and validation commands:

```bash
python scripts/run_robustness.py --config config/standard.json --source-mode onet31 --onet data/input/onet31.csv.zip
python scripts/sweep_spsa.py --config config/standard.json --source-mode onet31 --onet data/input/onet31.csv.zip
python scripts/check_results.py --results output/executed_demo
python tests/test_assignment.py
```

Install `matplotlib` for plotting, then run `python scripts/plot_results.py` to generate figures from the designated output directories.

## 8. File and directory guide

### Configuration and dependencies

| Path | Purpose |
| --- | --- |
| `requirements.txt` | Declares the NumPy, SciPy, and PyTorch package dependencies. |
| `config/standard.json` | Defines the full-size market, network, training, and solver configuration. |
| `config/quick.json` | Defines a compact configuration for shorter execution budgets. |
| `config/data_sources.json` | Records public data-source URLs, expected source files, and source metadata. |
| `data/` | Holds locally downloaded public data files. |
| `input/` | Provides an additional directory for locally supplied input files. |

### Core Python modules

| Path | Purpose |
| --- | --- |
| `gnn_bdge/__init__.py` | Marks the graph-matching implementation as an importable Python package. |
| `gnn_bdge/data.py` | Loads occupation skill profiles and constructs capacity-constrained market instances. |
| `gnn_bdge/model.py` | Defines the heterogeneous attention encoder and the GNN and MLP utility heads. |
| `gnn_bdge/training.py` | Runs utility prediction training and SPSA decision-feedback updates. |
| `gnn_bdge/solvers.py` | Implements LP/MIP optimization, matching baselines, feasibility checks, and solver event logging. |
| `gnn_bdge/experiments.py` | Coordinates market-level runs and saves model, assignment, and metric records. |

### Command-line scripts and tests

| Path | Purpose |
| --- | --- |
| `scripts/download_sources.py` | Downloads the supported public data files and records checksums. |
| `scripts/check_environment.py` | Captures Python, library, hardware, and solver availability information. |
| `scripts/run_experiments.py` | Launches the main matching experiments from a selected configuration. |
| `scripts/run_robustness.py` | Evaluates matching behavior across demand and capacity distribution shifts. |
| `scripts/sweep_spsa.py` | Evaluates SPSA direction counts and perturbation sizes. |
| `scripts/check_results.py` | Checks saved allocation counts, feasibility flags, and quality calculations. |
| `scripts/plot_results.py` | Produces summary figures from stored CSV metrics. |
| `tests/test_assignment.py` | Tests capacity accounting and agreement between the LP and integer objectives on a small market. |

### Output directories and summary files

| Path | Purpose |
| --- | --- |
| `output/environment.json` | Stores the captured software and hardware environment information. |
| `output/experiment_inventory.json` | Indexes experiment groups, source details, and available artifacts. |
| `output/executed_demo/` | Contains saved market-level and seed-level assignment experiments. |
| `output/non_graph_control/` | Contains the independent-utility control experiment outputs. |
| `output/robustness/` | Contains the distribution-shift metrics and HiGHS event logs. |
| `output/spsa_sensitivity/` | Contains SPSA parameter-sweep metrics and settings. |
| `output/figures/` | Contains summary charts in PNG format. |
| `output/*/run_details.json` | Records the input mode, configuration, environment, and run metadata for an experiment group. |
| `output/*/raw_results_per_market.csv` | Stores metrics for every recorded method and held-out market instance. |
| `output/*/raw_results_per_seed.csv` | Stores method-level metrics averaged within each seed. |
| `output/*/summary_by_method.csv` | Stores mean and sample-standard-deviation metrics by method and market size. |
| `output/*/training_by_seed.csv` | Stores offline training duration and step counts by seed. |
| `output/*/diagnostics.csv` | Stores optimization consistency and prediction-error diagnostic records. |
| `output/robustness/robustness_per_seed.csv` | Stores individual distribution-shift evaluation records. |
| `output/robustness/robustness_summary.csv` | Stores summary metrics at each distribution-shift level. |
| `output/spsa_sensitivity/spsa_parameter_sweep.csv` | Stores per-seed SPSA gradient and decision-loss observations. |
| `output/spsa_sensitivity/spsa_parameter_summary.csv` | Stores grouped mean and standard-deviation statistics for SPSA settings. |

### Files within a seed directory

Each `output/<experiment_group>/seeds/N<size>_seed<seed>/` directory groups the files associated with one market size and random seed.

| Filename or pattern | Purpose |
| --- | --- |
| `market_instance.npz` | Stores the first held-out market's numerical arrays, capacities, candidate edges, and utilities. |
| `market_edges.csv` | Lists candidate worker–job pairs and their actual utilities for the first held-out market. |
| `job_capacities.csv` | Lists the capacity assigned to each job in the first held-out market. |
| `gnn_weights.pt` | Stores the trained graph network's PyTorch parameter state. |
| `mlp_weights.pt` | Stores the trained pairwise MLP's PyTorch parameter state. |
| `train_losses.csv` | Stores supervised utility-prediction losses by training epoch. |
| `spsa_trace.csv` | Stores directional-gradient norms and decision-loss observations by SPSA step. |
| `method_results.csv` | Stores method-level records for the seed's held-out markets. |
| `candidate_scores_test<t>.csv` | Lists actual utilities alongside GNN and MLP scores for test market `t`. |
| `<method>_test<t>_assignment.csv` | Lists the selected worker–job pairs and scores for a method and test market. |
| `HiGHS_test<t>.jsonl` | Stores timestamped HiGHS solver-start and solver-finish records for test market `t`. |
| `Gurobi_test<t>.log` | Stores the native Gurobi solver log when that backend is enabled. |

All metrics and configurations should be interpreted together with their corresponding `run_details.json` and seed-level records.
