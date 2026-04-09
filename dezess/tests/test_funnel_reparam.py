"""Tests for funnel reparameterization and block-Gibbs."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def test_identity_roundtrip():
    from dezess.transforms import identity

    t = identity()
    x = jnp.array([1.0, 2.0, 3.0])
    z = t.inverse(x)
    x_rec = t.forward(z)
    np.testing.assert_allclose(x_rec, x, atol=1e-14)
    assert t.log_det_jac(z) == 0.0
