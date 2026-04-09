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


def test_ncp_funnel_roundtrip():
    """forward(inverse(x)) == x for the funnel transform."""
    from dezess.transforms import non_centered_funnel

    t = non_centered_funnel(width_idx=0, offset_indices=[1, 2, 3, 4])

    for log_w in [-3.0, 0.0, 3.0, 6.0]:
        x = jnp.array([log_w, 0.5, -1.0, 2.0, -0.3])
        z = t.inverse(x)
        x_rec = t.forward(z)
        np.testing.assert_allclose(x_rec, x, atol=1e-12)

    x = jnp.array([2.0, 1.0, -1.0, 0.5, -0.5])
    z = t.inverse(x)
    assert z[0] == x[0]
    expected_z_offsets = x[1:] / jnp.exp(x[0] / 2.0)
    np.testing.assert_allclose(z[1:], expected_z_offsets, atol=1e-12)


def test_ncp_funnel_jacobian():
    """log_det_jac matches numerical finite-difference Jacobian."""
    from dezess.transforms import non_centered_funnel

    t = non_centered_funnel(width_idx=0, offset_indices=[1, 2, 3])
    z = jnp.array([2.0, 0.5, -0.3, 1.0])

    ldj_analytical = t.log_det_jac(z)

    jac = jax.jacobian(t.forward)(z)
    _, ldj_numerical = jnp.linalg.slogdet(jac)

    np.testing.assert_allclose(ldj_analytical, ldj_numerical, atol=1e-10)
