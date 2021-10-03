# Copyright 2021 Multiscale Modeling of Fluid Materials, TU Munich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines types special to jax.edited.bak or this library. """

from typing import Any, NamedTuple
from functools import partial

from jax import tree_util
from jax import flatten_util
import jax.numpy as jnp

Array = jnp.ndarray
PyTree = Any

class Tensor(NamedTuple):
  """Vector and matrix pytree-products.

  Attributes:
    ndim: Dimension of the pytree (1: vector, 2: matrix)
    tensor: Data of the pytree

  """
  ndim: int
  tensor: PyTree

def tensor_matmul(matrix: Tensor, vector: PyTree):
  """Matrix vector product with a tensor and a pytree.

  Distinguishes between full matrices and diagonal matrices.

  Args:
    matrix: Matrix in tensor format
    vector: PyTree, which is compatible to the tensor

  """
  if matrix.ndim == 0:
    return tree_scale(matrix.tensor, vector)
  elif matrix.ndim == 1:
    return tree_multiply(matrix.tensor, vector)
  elif matrix.ndim == 2:
    return tree_matmul(matrix.tensor, vector)
  else:
    raise NotImplementedError(f"Cannot multiply matrix with dimension "
                              f"{matrix.ndim}")

def tree_multiply(tree_a: PyTree, tree_b: PyTree) -> PyTree:
  """Maps elementwise product over two vectors.

  Args:
    a: First pytree
    b: Second pytree, must have the same shape as a

  Returns:
    Returns a PyTree obtained by an element-wise product of all PyTree leaves.

  """
  return tree_util.tree_map(jnp.multiply, tree_a, tree_b)


def tree_scale(alpha: Array, tree: PyTree) -> PyTree:
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
  return tree_scale_imp(tree)


def tree_add(tree_a: PyTree, tree_b: PyTree) -> PyTree:
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
  return tree_add_imp(tree_a, tree_b)


def tree_matmul(tree_mat: Array, tree_vec: PyTree):
  """Matrix tree product for LD on manifold.

  Arguments:
    tree_mat: Matrix to be multiplied with flattened tree
    tree_vec: Tree representing vector

  Returns:
    Returns the un-flattened product of the matrix and the flattened tree.
  """
  # Todo: Redefine without need for flatten util
  vec_flat, unravel_fn = flatten_util.ravel_pytree(tree_vec)
  return unravel_fn(jnp.matmul(tree_mat, vec_flat))

def tree_dot(tree_a: PyTree, tree_b: PyTree):
  """Scalar product of two pytrees.

  Args:
    tree_a: First pytree
    tree_b: Second pytree with same tree stree structure and leaf shape as
      tree_a

  Returns:
    Returns a scalar, which is the sum of the element-wise product of all leaves.

  """
  leaves_a = tree_util.tree_leaves(tree_a)
  leaves_b = tree_util.tree_leaves(tree_b)
  return sum((jnp.sum(jnp.multiply(a, b)) for a, b in zip(leaves_a, leaves_b)))
