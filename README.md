# MCBS Simulation Code

This repository contains the reference implementation of the Monte Carlo-Inspired Business Strategy (MCBS) framework used to produce the simulation results reported in the paper “Monte Carlo-Inspired Business Strategy (MCBS): A Framework for Startup Business Exploration.”

The code simulates opportunity-driven startup exploration under three generation scenarios — A_Stable, B_Noisy, and C_Rare — and compares the MCBS selection algorithm (Algorithm 1 in the paper) against three baseline strategies: Random, Uniform, and Greedy. Each experimental cell runs 500 independent replications with a horizon of T = 1000 decision steps, and the outputs feed directly into Tables 3–5 of the manuscript.

## Files

| File                 | Purpose                                                                                                                                                                                                                                                                                                                                                                                    |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `mcbs_simulation.py` | Core simulation module: opportunity profile generators, the three scenario definitions (A_Stable, B_Noisy, C_Rare), the MCBS algorithm, and the three baseline strategies.                                                                                                                                                                                                                 |
| `run_experiment.py`  | Runs the full experiment — 3 scenarios × 4 strategies × 500 independent runs, T = 1000 steps per run — and produces Table 3 (cumulative utility CU, its standard deviation SDCU, and opportunity identification accuracy OIA), Table 4 (Welch’s t-test and Cohen’s d for cumulative utility, MCBS vs. Greedy), and Table 5 (two-proportion z-test and Cohen’s h for OIA, MCBS vs. Greedy). |
| `requirements.txt`   | Python dependencies.                                                                                                                                                                                                                                                                                                                                                                       |
| `README.md`          | This file.                                                                                                                                                                                                                                                                                                                                                                                 |

## Usage

Install the dependencies and run the complete experiment:

```text
pip install -r requirements.txt
python run_experiment.py
```

The script writes the descriptive statistics and the statistical test results used in the paper to the console and to summary output files.

## Parameter rationale

Some simulation parameters are not specified numerically in the manuscript. The values used in this repository therefore represent assumed parameter settings selected by the authors for the computational experiments.

### Utility weights

The utility function uses α = 2.0, β = 0.3, γ = 0.3, δ = 0.2, and λ = 0.2. These values were selected to assign greater importance to profit while retaining contributions from the associated cost and risk components.

### Softmax temperature

The softmax temperature is set to η = 8.0. This value was selected to provide a balance between exploration and exploitation in the MCBS selection rule.

### Opportunity attribute distributions

The per-attribute distributions follow the qualitative profiles described in Table 1 of the paper. The specific means and variances used for each opportunity archetype (e.g., `high_potential_low_sigma`, `red_herring`) are defined in the profile-generator functions in `mcbs_simulation.py`.

## Reproducibility

All random-number generation uses NumPy's `default_rng` with deterministic seeding per (scenario, strategy, run) combination. Re-running the experiment therefore reproduces the reported results exactly. If you change the seed scheme, scenario parameters, number of runs, or horizon, the resulting statistics will differ from those in the manuscript.
