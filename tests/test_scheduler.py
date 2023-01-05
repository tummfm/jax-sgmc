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

import itertools

import jax.numpy as jnp
from jax import random

import numpy as onp

import pytest

from jax_sgmc import scheduler
from jax_sgmc.util import testing


class TestScheduler:
  step_size = [None,
               (scheduler.polynomial_step_size,
                {"a": 1.0, "b": 1.0, "gamma": 0.55}),
               (scheduler.polynomial_step_size_first_last,
                {"first": 1.0, "last": 0.1, "gamma": 0.35})]

  burn_in = [None,
             (scheduler.initial_burn_in,
              {"n": 100})]

  temperature = [None,
                 (scheduler.constant_temperature,
                  {"tau": 0.5})]

  @pytest.mark.parametrize(
    "step_size, burn_in, temperature",
    itertools.product(step_size, burn_in, temperature))
  def test_scheduler(self, step_size, burn_in, temperature):
    """Test that the scheduler can initialize all specific schedulers. Moreover,
    the capability to provide default values is tested. """

    # Initialize all the specific schedulers
    if step_size is not None:
      fun, kwargs = step_size
      step_size = fun(**kwargs)
    if burn_in is not None:
      fun, kwargs = burn_in
      burn_in = fun(**kwargs)
    if temperature is not None:
      fun, kwargs = temperature
      temperature = fun(**kwargs)

    # Initialize the specific scheduler

    schedule = scheduler.init_scheduler(step_size=step_size,
                                        burn_in=burn_in,
                                        temperature=temperature)

    iterations = 100
    state, _ = schedule[0](iterations)

    for _ in range(iterations):

      sched = schedule[2](state)
      state = schedule[1](state)

      assert sched.step_size.shape == tuple()
      assert sched.temperature.shape == tuple()
      assert sched.burn_in.shape == tuple()
      assert sched.accept


class TestStepSize():

  @pytest.mark.parametrize("first, last, gamma, iterations",
                          itertools.product([1.0, 0.05],
                                            [0.01, 0.0009],
                                            [0.33, 0.55],
                                            [100, 14723]))
  def test_first_last(self, first, last, gamma, iterations):
    """Test, that the first and last step size are computed right."""

    first = jnp.array(first)
    last = jnp.array(last)
    gamma = jnp.array(gamma)

    schedule = scheduler.polynomial_step_size_first_last(first=first,
                                                         last=last,
                                                         gamma=gamma)

    state = schedule.init(iterations)

    testing.assert_close(schedule.get(state, 0), first)
    testing.assert_close(schedule.get(state, iterations-1), last)


class TestBurnIn():

  @pytest.mark.parametrize("n", [123, 243])
  def test_initial_burn_in(self, n):
    """Test, that no off by one error exists."""
    burn_in = scheduler.initial_burn_in(n=n)

    state, _ = burn_in.init(1000)

    # Check that samples are not accepted
    for idx in range(n):
      bi = burn_in.get(state, idx)
      state = burn_in.update(state, idx)
      assert bi == 0.0

    # Check that next sample is accepted
    assert burn_in.get(state, n + 1) == 1.0

class TestThinning():

  @pytest.fixture(params=[100, 1000])
  def burn_in(self, request):
    iterations = request.param
    accepted = random.bernoulli(random.PRNGKey(0), p=0.3, shape=(100,))
    init = lambda *args: (None, onp.sum(accepted))
    update = lambda *args, **kwargs: None
    get = lambda _, iteration, **kwargs: accepted[iteration]
    nonzero, = onp.nonzero(accepted)
    return scheduler.specific_scheduler(init, update, get), nonzero, iterations

  def test_random_thinning(self, burn_in):
    """Given a burn in schedule, the sample can only be accepted if it is not
    subject to burn in."""

    burn_in, non_zero, iterations = burn_in
    step_size = scheduler.polynomial_step_size(a=1.0, b=1.0, gamma=1.0)

    thinning = scheduler.random_thinning(
      step_size_schedule=step_size,
      burn_in_schedule=burn_in,
      selections=int(0.5 * non_zero.size))
    state, _ = thinning.init(iterations)

    accepted = 0
    for idx in range(iterations):
      # If the state is accepted, it must also be not subject to burn in
      if thinning.get(state, idx):
        assert (idx in non_zero)
        accepted += 1
    assert accepted == int(0.5 * non_zero.size)