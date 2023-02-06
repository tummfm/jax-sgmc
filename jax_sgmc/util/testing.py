"""Testing utility."""
from functools import partial

import numpy as onp
from numpy import testing

from jax import tree_util

assert_equal = partial(tree_util.tree_map, testing.assert_array_equal)


def assert_close(x, y, **kwargs):
  if "rtol" not in kwargs.keys():
    kwargs["rtol"] = 1e-5
  def assert_fn(xi, yi):
    xi = onp.ravel(xi)
    yi = onp.ravel(yi)
    testing.assert_allclose(xi, yi, **kwargs)
  tree_util.tree_map(assert_fn, x, y)
