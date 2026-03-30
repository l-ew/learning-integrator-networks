# Path Integration in Ring Attractor Networks

This codebase implements and analyzes recurrent neural networks that perform **path integration** — continuously tracking angular position by integrating a velocity input signal. The networks are based on ring attractor dynamics, where a localized bump of neural activity encodes the current head direction and is shifted by velocity-dependent control inputs.

## Installation

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate learning-integrator-networks
```


## Usage

### 1. Fit readout weights

```bash
python learn_networks.py --alpha 1e-4 --save-traces
```

Options:
- `--alpha`: regularization parameter(s) for ridge regression (default: `1e-4`, accepts multiple values)
- `--vary`: sweep over `sigma` or `N`
- `--save-traces` / `--discard-traces`: save or discard full driven network activity traces

### 2. Simulate closed-loop system

```bash
python simulate_closed_loop.py --set 0 --alpha 1e-4 --modes 1 --save-act
```

Options:
- `--set`: index of the training velocity set (default: `0`)
- `--alpha`: regularization parameter (default: `1e-4`)
- `--vary`: sweep over `sigma`, `g`, or `N`
- `--modes`: Fourier modes to include in the readout (default: all)
- `--save-act` / `--discard-act`: save or discard closed-loop activity traces

### 3. Stability analysis

```bash
python compute_floquet_multipliers.py --set 0 --alpha 1e-4
```

Computes Floquet multipliers and dominant eigenvectors of the monodromy matrix for a range of velocities.

### 4. Analyses and plotting

```bash
# Driven network bump shape
python plot_driven_network.py --set 0 --alpha 1e-4 --sim 0

# Analyze driven network
python analyze_driven_network.py --alpha 1e-4 --sim 0

# RMSE comparison between closed-loop and driven activity
python compare_driven_to_closed_loop.py --sim 0 --alpha 1e-4 --vel 360

# Compute theoretical solution
python theoretical_solution.py --alpha 1e-4 --vel 360

# Compare theory to simulation
python compare_theory_simulation.py --sim 0 --alpha 1e-4 --vel 360

# Closed-loop analysis (single sigma)
python analyze_closed_loop_single_sigma.py --set 0 --alpha 1e-4

# Closed-loop analysis (sigma sweep)
python analyze_closed_loop_sigma_sweep.py --set 0 --alpha 1e-4 --vary sigma --vals 50 100 150 200 250
```

## Project Structure

```
├── config.py                        # Simulation parameters
├── network.py                       # Network architectures and weight generation
├── utils.py                         # Shared utilities
├── environment.yml                  # Conda environment specification
├── training_vels.csv                # Training velocity sets
│
│  # Simulation
├── learn_networks.py                # Driven network simulation and readout fitting
├── simulate_closed_loop.py          # Closed-loop evaluation
├── online_learning.py               # Online learning variant
│
│  # Theory
├── theory.py                        # Continuum theory
├── theoretical_solution.py          # Analytical solutions
├── closed_loop_fixed_point.py       # Fixed-point analysis of the closed-loop system
├── compute_floquet_multipliers.py   # Floquet stability analysis
├── closed_loop_stability.py         # Closed-loop stability analysis
│
│  # Analysis
├── analyze_driven_network.py        # Driven network analysis
├── analyze_closed_loop_single_sigma.py  # Closed-loop analysis (single sigma)
├── analyze_closed_loop_sigma_sweep.py   # Closed-loop analysis (sigma sweep)
├── compare_driven_to_closed_loop.py # RMSE comparison between driven and closed-loop
├── compare_readout_weights.py       # Readout weight comparison
├── compare_readout_error.py         # Readout error comparison
├── compare_theory_simulation.py     # Theory vs. simulation comparison
│
│  # Plotting
├── plot_driven_network.py           # Driven network bump shape visualization
├── plot_cv_results.py               # Cross-validation results
├── visualization.py                 # Shared plotting utilities
└── matplotlib_config.py             # Matplotlib style configuration
```

## Configuration

All simulation parameters are defined in `config.py`, including network size (`N`), connectivity strength (`J0`), time step (`dt`), input structure, nonlinearity, number of Fourier modes, and the set of test velocities. Training velocity sets are specified in `training_vels.csv`. Parameters can also be swept from the command line using the `--vary` flag in the relevant scripts.