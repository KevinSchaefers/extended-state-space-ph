"""Reference solutions and error measures used in Section 4."""

import numpy as np
from scipy.integrate import solve_ivp

REF_RTOL = 2.5e-14
REF_ATOL = 1e-16


def reference_solution(system, x0, tspan, rtol=REF_RTOL, atol=REF_ATOL):
    """Reference solution computed with the adaptive eighth-order method ``DOP853``.

    Returns
    -------
    scipy.integrate.OdeSolution wrapper
        Result of ``solve_ivp`` with ``dense_output=True``.
    """
    return solve_ivp(
        system.rhs, tspan, x0, method="DOP853", dense_output=True, rtol=rtol, atol=atol
    )


def discrete_l2_error(sol, t, x):
    """Discrete ``L^2`` error of ``x`` on the equidistant time grid ``t``."""
    h = t[1] - t[0]
    return np.sqrt(h * np.sum(np.linalg.norm(sol.sol(t) - x, axis=0) ** 2))


def hamiltonian_or_nan(system, x):
    """Hamiltonian of ``system`` at ``x``, or ``NaN`` if ``x`` leaves the state space."""
    return system.H(x) if np.all(x > 0) else np.nan
