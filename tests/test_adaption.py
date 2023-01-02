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

import jax.numpy as jnp
from jax import random
from jax import flatten_util

import pytest

from jax_sgmc import adaption
from jax_sgmc import util
from jax_sgmc.util import testing

class TestDecorator:
  """Test that the decorator transforms array adaption to tree adaption."""

  @pytest.fixture
  def random_tree(self):
    key = random.PRNGKey(0)
    split1, split2 = random.split(key)
    tree = {"a": random.normal(split1, shape=(2,)),
            "b": {"b1": jnp.array(0.0),
                  "b2": random.normal(split2, shape=(3, 4))}}

    return tree


  @pytest.mark.parametrize("test_arg, test_kwarg", [(0.0, 1.0),
                                                    (1.0, 2.0),
                                                    (1.0, 1.0)])
  def test_args_kwargs(self, random_tree, test_arg, test_kwarg):

    @adaption.adaption(adaption.Manifold)
    def init_adaption():
      def init(sample, arg, kwarg=1.0):
        return sample, arg, kwarg
      def update(state, sample, sample_grad, arg, kwarg=1.0):
        return state, sample, sample_grad, arg, kwarg
      def get(state, sample, sample_grad, arg, kwarg=1.0):
        return arg * sample, kwarg * sample_grad, sample-sample_grad
      return init, update, get

    init, update, get = init_adaption()
    init_state = init(random_tree, test_arg, kwarg=test_kwarg)
    update_state = update(init_state, random_tree, random_tree, arg=test_arg, kwarg=test_kwarg)
    manifold = get(init_state, random_tree, random_tree, arg=test_arg, kwarg=test_kwarg)

    # Assert that parameters are passed correctly

    testing.assert_equal(init_state.ravel_fn(random_tree), init_state.state[0])
    assert init_state.state[1] == test_arg
    assert init_state.state[2] == test_kwarg

    testing.assert_equal(update_state.ravel_fn(random_tree), update_state.state[1])
    testing.assert_equal(update_state.ravel_fn(random_tree), update_state.state[2])
    assert update_state.state[3] == test_arg
    assert update_state.state[4] == test_kwarg

    assert manifold.g_inv.ndim == 1
    assert manifold.sqrt_g_inv.ndim == 1
    assert manifold.gamma.ndim == 1
    testing.assert_equal(manifold.g_inv.tensor, util.tree_scale(test_arg, random_tree))
    testing.assert_equal(manifold.sqrt_g_inv.tensor, util.tree_scale(test_kwarg, random_tree))
    testing.assert_equal(manifold.gamma.tensor, util.tree_scale(0.0, random_tree))

  @pytest.mark.parametrize("diag", [True, False])
  def test_diag_full(self, diag, random_tree):
    @adaption.adaption(adaption.Manifold)
    def init_adaption():
      def init(*args):
        return None
      def update(*args):
        return None
      def get(state, sample, *args, diag=True):
        if diag:
          return jnp.ones_like(sample), jnp.ones_like(sample), jnp.ones_like(sample)
        else:
          return jnp.eye(sample.size), jnp.eye(sample.size), jnp.ones_like(sample)
      return init, update, get
    init, _, get = init_adaption()

    init_state = init(random_tree)
    manifold = get(init_state, random_tree, random_tree, None, diag=diag)

    # Assert correctness of diagonal / full manifold
    if diag:
      assert manifold.g_inv.ndim == 1
      testing.assert_equal(util.tree_multiply(manifold.g_inv.tensor, random_tree), random_tree)
    else:
      assert manifold.g_inv.ndim == 2
      testing.assert_equal(util.tree_matmul(manifold.g_inv.tensor, random_tree), random_tree)
