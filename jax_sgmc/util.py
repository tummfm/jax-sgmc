"""Defines types special to jax or this library. """

import os
import re

from typing import Any
from functools import partial

from jax import tree_util
import jax.numpy as jnp

Array = jnp.ndarray
PyTree = Any

# Probably not important

def tree_multiply(a: PyTree, b: PyTree) -> PyTree:
  """Maps elementwise product over two vectors.

  Args:
    a: First pytree
    b: Second pytree, must have the same shape as a

  Returns:
    Returns a PyTree obtained by an element-wise product of all PyTree leaves.

  """
  return tree_util.tree_map(jnp.multiply, a, b)


def tree_scale(alpha: Array, a: PyTree) -> PyTree:
  """Scalar-Pytree product via tree_map.

  Args:
    alpha: Scalar
    a: Arbitrary PyTree

  Returns:
    Returns a PyTree with all leaves scaled by alpha.

  """
  @partial(partial, tree_util.tree_map)
  def tree_scale_imp(x: PyTree):
    return alpha * x
  return tree_scale_imp(a)


def tree_add(a: PyTree, b: PyTree) -> PyTree:
  """Maps elementwise sum over PyTrees.

  Arguments:
    a: First PyTree
    b: Second PyTree with the same shape as a

  Returns:
    Returns a PyTree obtained by leave-wise summation.
  """
  @partial(partial, tree_util.tree_map)
  def tree_add_imp(leaf_a, leaf_b):
    return leaf_a + leaf_b
  return tree_add_imp(a, b)

def pytree_list_transform(pytrees):

  treedef_list = tree_util.tree_structure([0 for _ in pytrees])
  treedef_single = tree_util.tree_structure(pytrees[0])

  def pytree_list_to_leaves(pytrees):
    """Transform a list of pytrees to allow pmap/vmap."""

    # Transpose the pytress, i. e. make a list (array) of leaves from a list of
    # pytrees. Only then vmap can be used to vectorize an operation over pytrees
    @partial(partial, tree_util.tree_map)
    def concatenate_leaves(*leaves):
      return jnp.concatenate([jnp.array([leaf]) for leaf in leaves], axis=0)

    return concatenate_leaves(*pytrees)

  def pytree_leaves_to_list(pytree):
    """Backtransform a pytree form pytree_list_to_leaves."""

    # We need to undo the concatenation along the first dimension
    @partial(partial, tree_util.tree_map)
    def split_leaves(leaf):
      return jnp.split(leaf, leaf.shape[0], axis=0)
    pytree_with_lists = split_leaves(pytree)

    # The pytree must be transposed
    pytrees = tree_util.tree_transpose(treedef_single,
                                       treedef_list,
                                       pytree_with_lists)

    # Each leaf is wrapped in an uneccessary list of length 1
    @partial(partial, tree_util.tree_map)
    def unlist(leaf):
      return leaf[0]

    return unlist(pytrees)

  return pytree_list_to_leaves, pytree_leaves_to_list


# Reproduced from numpyro
def set_host_device_count(n):
  """
  By default, XLA considers all CPU cores as one device. This utility tells XLA
  that there are `n` host (CPU) devices available to use. As a consequence, this
  allows parallel mapping in JAX :func:`jax.pmap` to work in CPU platform.

  .. note:: This utility only takes effect at the beginning of your program.
      Under the hood, this sets the environment variable
      `XLA_FLAGS=--xla_force_host_platform_device_count=[num_devices]`, where
      `[num_device]` is the desired number of CPU devices `n`.

  .. warning:: Our understanding of the side effects of using the
      `xla_force_host_platform_device_count` flag in XLA is incomplete. If you
      observe some strange phenomenon when using this utility, please let us
      know through our issue or forum page. More information is available in this
      `JAX issue <https://github.com/google/jax/issues/1408>`_.

  :param int n: number of CPU devices to use.
  """
  xla_flags = os.getenv('XLA_FLAGS', '').lstrip('--')
  xla_flags = re.sub(r'xla_force_host_platform_device_count=.+\s', '',
                     xla_flags).split()
  os.environ['XLA_FLAGS'] = ' '.join(
    ['--xla_force_host_platform_device_count={}'.format(n)]
    + xla_flags)
