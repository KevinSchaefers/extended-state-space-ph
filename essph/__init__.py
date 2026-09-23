"""Reference implementation for the numerical experiments of Section 4 of

K. Schaefers, A. Bartel, M. Guenther:
Energy-Consistent Splitting Methods for Port-Hamiltonian ODEs via a Generalized
Extended State Space Approach.
"""

from .integrators import (
    SolverStats,
    ead,
    ess,
    ess_strang_dg,
    ess_strang_dg_one_sided,
    gl1,
    gl2,
    gonzalez_discrete_gradient,
    simplified_newton,
    triple_jump_coefficients,
)
from .systems import (
    R_hat,
    SkewGradientSystem,
    current_source,
    pHSystem,
    setup_lc_chain,
    setup_lc_chain_benchmark,
    setup_lotka_volterra,
)
from .utils import discrete_l2_error, hamiltonian_or_nan, reference_solution

__all__ = [
    "SkewGradientSystem",
    "pHSystem",
    "setup_lotka_volterra",
    "setup_lc_chain",
    "setup_lc_chain_benchmark",
    "current_source",
    "R_hat",
    "gonzalez_discrete_gradient",
    "triple_jump_coefficients",
    "ess_strang_dg",
    "ess_strang_dg_one_sided",
    "simplified_newton",
    "gl1",
    "gl2",
    "ead",
    "ess",
    "SolverStats",
    "reference_solution",
    "discrete_l2_error",
    "hamiltonian_or_nan",
]
