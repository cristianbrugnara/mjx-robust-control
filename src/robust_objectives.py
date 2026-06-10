"""
Robust training objectives. Each function reduces a 1D array of per-rollout costs to a scalar
that can be differentiated through. Supported: mean (empirical average), cvar (Rockafellar-
Uryasev surrogate with trainable threshold tau), pinball (quantile surrogate), softmax
(log-sum-exp worst-case), worst_case (empirical maximum).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jax.Array


def mean_loss(costs: Array) -> Array:
    """Empirical average rollout cost."""
    costs = jnp.asarray(costs)
    return jnp.mean(costs)


def cvar_loss(costs: Array, alpha: float, tau: Array) -> Array:
    """Rockafellar-Uryasev CVaR surrogate.

    Parameters
    ----------
    costs:
        1D array of rollout costs.
    alpha:
        Tail probability in (0, 1]. This matches the notebook's
        `cvar_objective(psi_vals, tau, alpha_train)`.
    tau:
        Trainable threshold parameter.
    """
    costs = jnp.asarray(costs)
    tau = jnp.asarray(tau, dtype=costs.dtype)
    alpha = jnp.clip(jnp.asarray(alpha, dtype=costs.dtype), 1.0e-6, 1.0)
    return tau + jnp.mean(jax.nn.relu(costs - tau)) / alpha


def pinball_loss(costs: Array, quantile: float, tau: Array) -> Array:
    """pinball / quantile surrogate.

    Parameters
    ----------
    quantile:
        Target quantile q in [0, 1]. The notebook uses q = 1 - alpha.
    tau:
        Trainable threshold parameter.
    """
    costs = jnp.asarray(costs)
    tau = jnp.asarray(tau, dtype=costs.dtype)
    q = jnp.clip(jnp.asarray(quantile, dtype=costs.dtype), 0.0, 1.0)
    r = costs - tau
    return jnp.mean(q * jax.nn.relu(r) + (1.0 - q) * jax.nn.relu(-r))


def softmax_loss(costs: Array, beta: float) -> Array:
    """Smooth worst-case objective."""
    costs = jnp.asarray(costs)
    beta = jnp.asarray(beta, dtype=costs.dtype)

    def _logmeanexp() -> Array:
        z = beta * costs
        z_max = jnp.max(z)
        return (jnp.log(jnp.mean(jnp.exp(z - z_max))) + z_max) / beta

    return jnp.where(jnp.abs(beta) < 1.0e-8, jnp.mean(costs), _logmeanexp())


def worst_case_loss(costs: Array) -> Array:
    """Empirical maximum rollout cost."""
    costs = jnp.asarray(costs)
    return jnp.max(costs)


def objective_requires_tau(name: str) -> bool:
    return name in ("cvar", "pinball")


__all__ = [
    "mean_loss",
    "cvar_loss",
    "pinball_loss",
    "softmax_loss",
    "worst_case_loss",
    "objective_requires_tau",
]
