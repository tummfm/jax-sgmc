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

"""Utility to evaluate stochastic or true potential.

This module transforms the likelihood function for a single observation or a
batch of observations to a function calculating the stochastic or full potential
making use of ``map``, ``vmap`` and ``pmap``.

"""

from functools import partial

from typing import Callable, Any, AnyStr, Optional, Tuple, Union, Protocol

from jax import vmap, pmap, lax, tree_util, named_call

import jax.numpy as jnp

from jax_sgmc import util
from jax_sgmc.data import CacheState, MiniBatch

PyTree = Any
Array = util.Array

Likelihood = Union[
  Callable[[PyTree, PyTree, MiniBatch], Tuple[Array, PyTree]],
  Callable[[PyTree, MiniBatch], Array]]
Prior = Callable[[PyTree], Array]

class StochasticPotential(Protocol):
  def __call__(self,
               sample: PyTree,
               reference_data: MiniBatch,
               state: PyTree = None,
               mask: Array = None,
               likelihoods: bool = False
               ) -> Union[Tuple[Array, PyTree],
                          Tuple[Array, Tuple[Array, PyTree]]]:
    """Calculates the stochastic potential for a mini-batch of data.

      Args:
        sample: Model parameters
        reference_data: Batch of observations
        state: Special parameters of the model which should not change the
          result of a model evaluation.
        mask: Marking invalid (e.g. double) samples
        likelihoods: Return the likelihoods of all model evaluations separately

      Returns:
        Returns an approximation of the true potential based on a mini-batch of
        reference data. Moreover, the likelihood for every single observation
        can be returned.

     """

class FullPotential(Protocol):
  def __call__(self,
               sample: PyTree,
               data_state: CacheState,
               full_data_map_fn: Callable,
               state: PyTree = None
               ) -> Tuple[Array, Tuple[CacheState, PyTree]]:
    """Calculates the potential over the full dataset.

      Args:
        sample: Model parameters
        data_state: State of the ``full_data_map`` functional
        full_data_map_fn: Functional mapping a function over the complete
          dataset
        state: Special parameters of the model which should not change the
          result of a model evaluation.

      Returns:
        Returns the potential of the current sample using the full dataset.

    """



# Todo: Possibly support soft-vmap (numpyro)

def minibatch_potential(prior: Prior,
                        likelihood: Callable,
                        strategy: AnyStr = "map",
                        has_state: bool = False,
                        is_batched: bool = False,
                        temperature: float = 1.) -> StochasticPotential:
  """Initializes the potential function for a minibatch of data.

  Args:
    prior: Log-prior function which is evaluated for a single
      sample.
    likelihood: Log-likelihood function. If ``has_state = True``, then the
      first argument is the model state, otherwise the arguments are ``sample,
      reference_data``.
    strategy: Determines hwo to evaluate the model function with respect for
      sample:

      - ``'map'`` sequential evaluation
      - ``'vmap'`` parallel evaluation via vectorization
      - ``'pmap'`` parallel evaluation on multiple devices

    has_state: If an additional state is provided for the model evaluation
    is_batched: If likelihood expects a batch of observations instead of a
      single observation. If the likelihood is batched, choosing the strategy
      has no influence on the computation.
    temperature: Posterior temperature. T = 1 is the Bayesian posterior.

  Returns:
    Returns a function which evaluates the stochastic potential for a mini-batch
    of data. The first argument are the latent variables and the second is the
    mini-batch.
  """

  # State is always passed to simplify usage in solvers
  def stateful_likelihood(state: PyTree,
                          sample: PyTree,
                          reference_data: PyTree):
    if has_state:
      lks, state = likelihood(state, sample, reference_data)
    else:
      lks = likelihood(sample, reference_data)
      state = None
    # Ensure that a scalar is returned to avoid broadcasting with mask
    return jnp.squeeze(lks), state

  # Define the strategies to evaluate the likelihoods sequantially, vectorized
  # or in parallel
  if is_batched:
    batched_likelihood = stateful_likelihood
  elif strategy == 'map':
    def batched_likelihood(state: PyTree,
                           sample: PyTree,
                           reference_data: PyTree):
      partial_likelihood = partial(stateful_likelihood, state, sample)
      return lax.map(partial_likelihood, reference_data)
  elif strategy == 'pmap':
    batched_likelihood = pmap(stateful_likelihood,
                              in_axes=(None, None, 0))
  elif strategy == 'vmap':
    batched_likelihood = vmap(stateful_likelihood,
                              in_axes=(None, None, 0))
  else:
    raise NotImplementedError(f"Strategy {strategy} is unknown")


  def batch_potential(state: PyTree,
                      sample: PyTree,
                      reference_data: MiniBatch,
                      mask: Array):
    # Approximate the potential by taking the average and scaling it to the
    # full data set size
    batch_data, batch_information = reference_data
    N = batch_information.observation_count
    n = batch_information.batch_size

    batch_likelihoods, new_states = batched_likelihood(
      state, sample, batch_data)
    if is_batched:
      # Batched evaluation returns single state
      new_state = new_states
    elif state is not None:
      new_state = tree_util.tree_map(
        lambda ary, org: jnp.reshape(jnp.take(ary, 0, axis=0), org.shape),
        new_states, state)
    else:
      new_state = None

    # The mask is only necessary for the full potential evaluation
    if mask is None:
      stochastic_potential = - N * jnp.mean(batch_likelihoods, axis=0)
    else:
      stochastic_potential = - N / n * jnp.dot(batch_likelihoods, mask)
    return stochastic_potential, batch_likelihoods, new_state

  @partial(named_call, name='evaluate_stochastic_potential')
  def potential_function(sample: PyTree,
                         reference_data: MiniBatch,
                         state: PyTree = None,
                         mask: Array = None,
                         likelihoods: bool = False):
    # Never differentiate w. r. t. reference data
    reference_data = lax.stop_gradient(reference_data)

    # Evaluate the likelihood and model for each reference data sample
    # likelihood_value = batched_likelihood_and_model(sample, reference_data)
    # It is also possible to combine the prior and the likelihood into a single
    # callable.

    batch_likelihood, observation_likelihoods, new_state = batch_potential(
      state, sample, reference_data, mask)

    # The prior has to be evaluated only once, therefore the extra call
    prior_value = prior(sample)

    if likelihoods:
      return (
        jnp.squeeze(batch_likelihood - prior_value) / temperature,
        (observation_likelihoods, new_state))
    else:
      return (jnp.squeeze(batch_likelihood - prior_value) / temperature,
              new_state)

  return potential_function


def full_potential(prior: Callable[[PyTree], Array],
                   likelihood: Callable[[PyTree, PyTree], Array],
                   strategy: AnyStr = "map",
                   has_state: bool = False,
                   is_batched: bool = False,
                   temperature: float = 1.,
                   ) -> FullPotential:
  """Transforms a pdf to compute the full potential over all reference data.

  Args:
    prior: Log-prior function which is evaluated for a single
      sample.
    likelihood: Log-likelihood function. If ``has_state = True``, then the
      first argument is the model state, otherwise the arguments are ``sample,
      reference_data``.
    strategy: Determines how to evaluate the model function with respect for
      sample:

      - ``'map'`` sequential evaluation
      - ``'vmap'`` parallel evaluation via vectorization
      - ``'pmap'`` parallel evaluation on multiple devices

    has_state: If an additional state is provided for the model evaluation
    is_batched: If likelihood expects a batch of observations instead of a
      single observation. If the likelihood is batched, choosing the strategy
      has no influence on the computation. In this case, the last argument of
      the likelihood should be an optional mask. The mask is an arrays with ones
      for valid observations and zeros for non-valid observations.
    temperature: Posterior temperature. T = 1 is the Bayesian posterior.

  Returns:
    Returns a function which evaluates the potential over the full dataset via
    a dataset mapping from the :mod:`jax_sgmc.data` module.

  """
  assert strategy != 'pmap', "Pmap is currently not supported"

  # Can use the potential evaluation strategy for a minibatch of data. The prior
  # must be evaluated independently.
  batch_potential = minibatch_potential(lambda _: jnp.array(0.0),
                                        likelihood,
                                        strategy=strategy,
                                        has_state=has_state,
                                        is_batched=is_batched)

  def batch_evaluation(sample, reference_data, mask, state):
    potential, state = batch_potential(sample, reference_data, state, mask)
    # We need to undo the scaling to get the real potential
    _, batch_information = reference_data
    N = batch_information.observation_count
    n = batch_information.batch_size
    unscaled_potential = potential * n / N
    return unscaled_potential, state

  @partial(named_call, name='evaluate_true_potential')
  def sum_batched_evaluations(sample: PyTree,
                              data_state: CacheState,
                              full_data_map_fn: Callable,
                              state: PyTree = None):
    body_fn = partial(batch_evaluation, sample)

    if data_state is None:  # quick fix to let it run with full_data_mapper
      results, new_state = full_data_map_fn(
        body_fn, state, masking=True, information=True)
    else:
      data_state, (results, new_state) = full_data_map_fn(
        body_fn, data_state, state, masking=True, information=True)

    # The prior needs just a single evaluation
    prior_eval = prior(sample)

    return (jnp.squeeze(jnp.sum(results) - prior_eval) / temperature,
            (data_state, new_state))

  return sum_batched_evaluations
