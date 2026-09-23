# Energy-consistent extended state space methods for port-Hamiltonian ODEs

Reference implementation of the numerical experiments in

> K. Schäfers, A. Bartel, M. Günther:
> *Energy-Consistent Splitting Methods for Port-Hamiltonian ODEs via a Generalized Extended State Space Approach*, 2026.

## Contents

```
essph/
  systems.py       system classes and the two benchmark problems
  integrators.py   time integration methods and effort statistics
  utils.py         reference solutions and error measures
notebooks/
  section_4-1_lotka_volterra.ipynb   Figures 1 and 2
  section_4-2_lc_chain.ipynb         Figures 4, 5 and 6, Table 1
```

| Notebook | Experiment | Output |
| --- | --- | --- |
| `section_4-1_lotka_volterra.ipynb` | convergence of the Strang splitting and its triple-jump compositions up to order 10 | Figure 1a |
| | energy behavior of both components of the extended state | Figure 1b |
| | one-sided vs. symmetric consistent approximations | Figures 2a and 2b |
| `section_4-2_lc_chain.ipynb` | reference solution of the LC-oscillator chain with 3 building blocks | Figure 4 |
| | convergence and computational effort for 50 building blocks | Figure 5, Table 1 |
| | energy behavior of the extended state space method | Figure 6 |

## Methods

| Function | Method |
| --- | --- |
| `ess_strang_dg` | extended state space method, symmetric consistent approximations (both copies are energy-consistent), Gonzalez discrete gradient method for the numerical sub-flows, Strang splitting and triple-jump compositions |
| `ess_strang_dg_one_sided` | extended state space method, one-sided consistent approximations (only first copy is energy-consistent) |
| `gl1`, `gl2` | 1-stage and 2-stage Gauss-Legendre methods |
| `ead` | Strang splitting based on the energy-associated decomposition. Energy-conservative sub-flow is computed exactly, passive sub-flow is computed numerically via 1-stage Gauss-Legendre method  |
| `ess` | extended state space method based on the Strang splitting and numerical sub-flows computed via the 1-stage Gauss-Legendre method |

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks
```

The results in the paper were obtained with Python 3.13.0.
A full run of `section_4-1_lotka_volterra.ipynb` and `section_4-2_lc_chain.ipynb` takes about five minutes each on a standard laptop.

## License

MIT, see `LICENSE`.
