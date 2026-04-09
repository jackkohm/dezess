"""Online diagnostics for monitoring sampler health during production.

Provides streaming estimates of ESS and R-hat that can be computed
incrementally without storing all samples. Used by the variant runner
to report convergence progress and support early stopping.
"""

from __future__ import annotations

import numpy as np


class StreamingDiagnostics:
    """Incrementally compute ESS and split R-hat from streaming samples.

    Maintains per-walker running statistics (mean, variance, lag-1
    autocovariance) and computes diagnostics from them.

    Parameters
    ----------
    n_walkers : int
        Number of walkers (treated as independent chains).
    n_dim : int
        Number of dimensions.
    """

    def __init__(self, n_walkers: int, n_dim: int):
        self.n_walkers = n_walkers
        self.n_dim = n_dim
        self.n_steps = 0
        # Per-walker, per-dim running stats (Welford's algorithm)
        self._mean = np.zeros((n_walkers, n_dim))
        self._m2 = np.zeros((n_walkers, n_dim))  # sum of squared deviations
        self._prev = np.zeros((n_walkers, n_dim))
        self._lag1_sum = np.zeros((n_walkers, n_dim))  # sum of (x_t - mean)(x_{t-1} - mean)
        # Divergence tracking
        self._n_divergent = 0
        self._max_abs_value = 0.0

    def update(self, positions: np.ndarray, log_probs: np.ndarray = None):
        """Update with new positions (n_walkers, n_dim).

        Call once per production step. Optionally pass log_probs
        (n_walkers,) for divergence detection.
        """
        self.n_steps += 1
        n = self.n_steps
        x = np.asarray(positions)

        # Track divergences: extreme values or NaN/Inf
        max_abs = float(np.max(np.abs(x)))
        if max_abs > self._max_abs_value:
            self._max_abs_value = max_abs
        if np.any(~np.isfinite(x)):
            self._n_divergent += 1
        if log_probs is not None:
            lp = np.asarray(log_probs)
            if np.any(lp < -1e29) or np.any(~np.isfinite(lp)):
                self._n_divergent += 1

        if n == 1:
            self._mean[:] = x
            self._prev[:] = x
            return

        old_mean = self._mean.copy()
        delta = x - self._mean
        self._mean += delta / n
        delta2 = x - self._mean
        self._m2 += delta * delta2

        # Lag-1 autocovariance (approximate)
        if n > 2:
            self._lag1_sum += (x - self._mean) * (self._prev - old_mean)

        self._prev[:] = x

    def ess_per_dim(self) -> np.ndarray:
        """Estimate ESS per dimension using lag-1 autocorrelation.

        Returns (n_dim,) array. Uses the simple IAT ≈ (1 + 2*rho_1) / (1 - 2*rho_1)
        approximation, which is fast but conservative.
        """
        if self.n_steps < 10:
            return np.zeros(self.n_dim)

        n = self.n_steps
        var = self._m2 / (n - 1)  # (n_walkers, n_dim)
        var = np.maximum(var, 1e-30)

        # Lag-1 autocorrelation per walker per dim
        rho1 = self._lag1_sum / ((n - 2) * var + 1e-30)
        rho1 = np.clip(rho1, -0.99, 0.99)

        # IAT ≈ (1 + rho1) / (1 - rho1) for AR(1)-like chains
        iat = np.maximum((1 + rho1) / (1 - rho1), 1.0)

        # ESS = n_steps * n_walkers / mean_IAT_across_walkers
        mean_iat = np.mean(iat, axis=0)
        ess = n * self.n_walkers / mean_iat
        return ess

    def ess_min(self) -> float:
        """Minimum ESS across dimensions."""
        ess = self.ess_per_dim()
        return float(np.min(ess)) if len(ess) > 0 else 0.0

    def rhat_per_dim(self) -> np.ndarray:
        """Simple R-hat estimate treating each walker as a chain.

        Returns (n_dim,) array. Uses the standard between/within chain
        variance ratio.
        """
        if self.n_steps < 4:
            return np.full(self.n_dim, np.inf)

        n = self.n_steps
        var = self._m2 / (n - 1)  # (n_walkers, n_dim)

        W = np.mean(var, axis=0)  # within-chain variance
        chain_means = self._mean  # (n_walkers, n_dim)
        B = np.var(chain_means, axis=0, ddof=1) * n  # between-chain variance

        W_safe = np.maximum(W, 1e-30)
        var_hat = (1 - 1.0/n) * W_safe + B / n
        rhat = np.sqrt(var_hat / W_safe)
        # Cap at reasonable values for early steps
        rhat = np.minimum(rhat, 100.0)
        return rhat

    def rhat_max(self) -> float:
        """Maximum R-hat across dimensions."""
        rhat = self.rhat_per_dim()
        return float(np.max(rhat)) if len(rhat) > 0 else np.inf

    def ensemble_diversity(self) -> float:
        """Ratio of between-walker variance to within-walker variance.

        Values near 0 indicate mode collapse (all walkers in the same place).
        Values near 1 indicate good diversity (walkers spread across the target).
        Values >> 1 may indicate walkers haven't converged to the same distribution.

        This is essentially (B/W) averaged across dimensions, where B is the
        between-chain variance and W is the within-chain variance.
        """
        if self.n_steps < 4:
            return 0.0

        n = self.n_steps
        var = self._m2 / (n - 1)  # (n_walkers, n_dim)
        W = np.mean(var, axis=0)  # within-chain variance per dim
        B = np.var(self._mean, axis=0, ddof=1)  # between-chain variance per dim

        W_safe = np.maximum(W, 1e-30)
        ratio = B / W_safe  # per-dim diversity ratio
        return float(np.mean(ratio))

    def summary(self) -> dict:
        """Return a summary dict of current diagnostics."""
        ess = self.ess_per_dim()
        rhat = self.rhat_per_dim()
        return {
            "n_steps": self.n_steps,
            "ess_min": float(np.min(ess)),
            "ess_mean": float(np.mean(ess)),
            "rhat_max": float(np.max(rhat)),
            "rhat_mean": float(np.mean(rhat)),
            "diversity": self.ensemble_diversity(),
            "n_divergent": self._n_divergent,
        }
