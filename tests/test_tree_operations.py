from functools import partial


import jax
import jax.numpy as jnp
from jax import random
from jax import flatten_util
from jax import tree_util

from jax_sgmc import util
from jax_sgmc import data
from jax_sgmc.util import testing

# Todo: Test vmap on custom host_callback

import pytest



@pytest.fixture
def random_tree():
  key = random.PRNGKey(0)
  split1, split2 = random.split(key)
  tree = {"a": random.normal(split1, shape=(2,)),
          "b": {"b1": 0.0,
                "b2": random.normal(split2, shape=(3, 4))}}

  flat_tree, unravel_fn = flatten_util.ravel_pytree(tree)
  ravel_fn = lambda t: flatten_util.ravel_pytree(t)[0]
  return (tree, flat_tree), (ravel_fn, unravel_fn)


class TestTree:

  def test_tree_scale(self, random_tree):
    (tree, flat_tree), (ravel_fn, unravel_fn) = random_tree
    alpha = random.normal(random.PRNGKey(1))

    true_result = unravel_fn(alpha * flat_tree)
    treemap_result = util.tree_scale(alpha, tree)

    testing.assert_equal(true_result, treemap_result)

  def test_tree_multiply(self, random_tree):
    (tree, flat_tree), (ravel_fn, unravel_fn) = random_tree
    rnd = random.normal(random.PRNGKey(1), shape=flat_tree.shape)

    true_result = unravel_fn(jnp.multiply(rnd, flat_tree))
    treemap_result = util.tree_multiply(unravel_fn(rnd), tree)

    testing.assert_equal(true_result, treemap_result)

  def test_tree_add(self, random_tree):
    (tree, flat_tree), (ravel_fn, unravel_fn) = random_tree
    rnd = random.normal(random.PRNGKey(1), shape=flat_tree.shape)

    true_result = unravel_fn(jnp.add(rnd, flat_tree))
    treemap_result = util.tree_add(unravel_fn(rnd), tree)

    testing.assert_equal(true_result, treemap_result)

  def test_tree_matmul(self, random_tree):
    (tree, flat_tree), (ravel_fn, unravel_fn) = random_tree
    rnd = random.normal(random.PRNGKey(1), shape=(flat_tree.size, flat_tree.size))

    true_result = unravel_fn(jnp.matmul(rnd, flat_tree))
    treemap_result = util.tree_matmul(rnd, tree)

    testing.assert_equal(true_result, treemap_result)

  def test_tree_dot(self, random_tree):
    (tree, flat_tree), (ravel_fn, unravel_fn) = random_tree

    true_result = jnp.dot(flat_tree, flat_tree)
    treemap_result = util.tree_dot(tree, tree)

    testing.assert_equal(true_result, treemap_result)

class TestTreeMap():

  def test_vmap(self, random_tree):
    (random_tree, _), _ = random_tree

    zero_tree = tree_util.tree_map(jnp.zeros_like, random_tree)
    one_tree = tree_util.tree_map(jnp.ones_like, random_tree)
    modified_tree = tree_util.tree_map(jnp.add, random_tree, one_tree)

    @jax.jit
    @util.list_vmap
    def test_substract(tree):
      return tree_util.tree_map(jnp.subtract, modified_tree, tree)

    one, zero = test_substract(random_tree, modified_tree)

    testing.assert_close(zero, zero_tree)
    testing.assert_close(one, one_tree)

  @pytest.mark.pmap
  def test_pmap(self, random_tree):
    (random_tree, _), _ = random_tree

    zero_tree = tree_util.tree_map(jnp.zeros_like, random_tree)
    one_tree = tree_util.tree_map(jnp.ones_like, random_tree)
    modified_tree = tree_util.tree_map(jnp.add, random_tree, one_tree)

    @jax.jit
    @util.list_pmap
    def test_substract(tree):
      return tree_util.tree_map(jnp.subtract, modified_tree, tree)

    one, zero = test_substract(random_tree, modified_tree)

    testing.assert_close(zero, zero_tree)
    testing.assert_close(one, one_tree)