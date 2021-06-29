from functools import partial, wraps

import jax
from jax import tree_util
import jax.numpy as jnp

def pytree_list_to_leaves(pytrees):
  """Transform a list of pytrees to allow pmap/vmap.

  Example usage:

  .. doctest::

    >>> import jax.numpy as jnp
    >>> import jax_sgmc.util.list_map as lm
    >>>
    >>> tree_a = {"a": 0.0, "b": jnp.zeros((2,))}
    >>> tree_b = {"a": 1.0, "b": jnp.ones((2,))}
    >>>
    >>> concat_tree = lm.pytree_list_to_leaves([tree_a, tree_b])
    >>> print(concat_tree)
    {'a': DeviceArray([0., 1.], dtype=float32), 'b': DeviceArray([[0., 0.],
                 [1., 1.]], dtype=float32)}


  Args:
    pytrees: A list of trees with similar tree structure and equally shaped
      leaves

  Returns:
    Returns a tree with the same tree structure but corresponding leaves
    concatenated along the first dimension.

  """

  # Transpose the pytress, i. e. make a list (array) of leaves from a list of
  # pytrees. Only then vmap can be used to vectorize an operation over pytrees
  @partial(partial, tree_util.tree_map)
  def concatenate_leaves(*leaves):
    return jnp.stack(leaves, axis=0)
  return concatenate_leaves(*pytrees)


def pytree_leaves_to_list(pytree):
  """Splits a pytree in a list of pytrees.

  Splits every leaf of the pytree along the first dimenion, thus undoing the
  :func:`pytree_list_to_leaves` transformation.

  Example usage:

  .. doctest::

    >>> import jax.numpy as jnp
    >>> import jax_sgmc.util.list_map as lm
    >>>
    >>> tree = {"a": jnp.array([0.0, 1.0]), "b": jnp.zeros((2, 2))}
    >>>
    >>> tree_list = lm.pytree_leaves_to_list(tree)
    >>> print(tree_list)
    [{'a': DeviceArray(0., dtype=float32), 'b': DeviceArray([0., 0.], dtype=float32)}, {'a': DeviceArray(1., dtype=float32), 'b': DeviceArray([0., 0.], dtype=float32)}]


  Args:
    pytree: A single pytree where each leaf has eqal `leaf.shape[0]`.

  Returns:
    Returns a list of pytrees with similar structure.

  """
  leaves, treedef = tree_util.tree_flatten(pytree)
  num_trees = leaves[0].shape[0]
  pytrees = [tree_util.tree_unflatten(treedef, [leaf[idx] for leaf in leaves])
             for idx in range(num_trees)]
  return pytrees


def list_vmap(fun):
  """vmaps a function over similar pytrees.

  Example usage:

  .. doctest::

    >>> from jax import tree_map
    >>> import jax.numpy as jnp
    >>> import jax_sgmc.util.list_map as lm
    >>>
    >>> tree_a = {"a": 0.0, "b": jnp.zeros((2,))}
    >>> tree_b = {"a": 1.0, "b": jnp.ones((2,))}
    >>>
    ... @lm.list_vmap
    ... def tree_add(pytree):
    ...   return tree_map(jnp.subtract, pytree, tree_b)
    >>>
    >>> print(tree_add(tree_a, tree_b))
    [{'a': DeviceArray(-1., dtype=float32), 'b': DeviceArray([-1., -1.], dtype=float32)}, {'a': DeviceArray(0., dtype=float32), 'b': DeviceArray([0., 0.], dtype=float32)}]

  Args:
    fun: Function accepting a single pytree as first argument.

  Returns:
    Returns a vmapped-function accepting multiple pytree args with similar tree-
    structure.

  """
  vmap_fun = jax.vmap(fun, 0, 0)
  @wraps(fun)
  def vmapped(*pytrees):
    single_tree = pytree_list_to_leaves(pytrees)
    print(single_tree)
    single_result = vmap_fun(single_tree)
    return pytree_leaves_to_list(single_result)
  return vmapped

def list_pmap(fun):
  """pmaps a function over similar pytrees.

  Args:
    fun: Function accepting a single pytree as first argument.

  Returns:
    Returns a pmapped-function accepting multiple pytree args with similar tree-
    structure.

  """
  pmap_fun = jax.pmap(fun, 0)
  @wraps(fun)
  def pmapped(*pytrees):
    single_tree = pytree_list_to_leaves(pytrees)
    single_result = pmap_fun(single_tree)
    return pytree_leaves_to_list(single_result)
  return pmapped
