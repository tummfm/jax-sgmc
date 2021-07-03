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

"""Utility to evaluate stochastic or real potential.

Stochastic gradient monte carlo requires to evaluate the potential and the model
for a multiple of observations or all observations. However, the likelihood and
model function only accept a singe observation and parameter set. Therefore,
this module maps the evaluation over the mini-batch or even all observations by
making use of jaxs tools ``map``, ``vmap`` and ``pmap``.

"""

# Todo: Usage example

from functools import partial

from typing import Callable, Any, AnyStr, Optional, Tuple, Union

from jax import vmap, pmap, lax, tree_util

import jax.numpy as jnp

from jax_sgmc import util
from jax_sgmc.data import CacheState, MiniBatch

# Here we define special types

PyTree = Any
Array = util.Array

Likelihood = Callable[[Optional[PyTree], PyTree, MiniBatch],
                      Union[Tuple[Array, PyTree], Array]]
Prior = Callable[[PyTree], Array]

StochasticPotential = Callable[[PyTree, MiniBatch], Tuple[Array, PyTree]]
FullPotential = Callable[[PyTree, CacheState], Tuple[Array, PyTree]]

# Todo: Possibly support soft-vmap (numpyro)
# Todo: Implement evaluation via pmap

def minibatch_potential(prior: Prior,
                        likelihood: Likelihood,
                        strategy: AnyStr = "map",
                        has_state = False,
                        is_batched = False) -> StochasticPotential:
  """Initializes the potential function for a minibatch of data.

  Args:
    prior: Probability density function which is evaluated for a single
      sample.
    likelihood: Probability density function. If ``has_state = True``, then the
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

  Returns:
    Returns a function which evaluates the stochastic potential for a mini-batch
    of data. The first argument are the latent variables and the second is the
    mini-batch.
  """

  # Todo: Keep the state only from the first evaluation or from any evaluation
  #       at random? -> Reference data is random, samples are the same.
  # The final function to evaluate the potential including likelihood and prio
  if is_batched:
    # The likelihood is already provided for a batch of data
    def batch_potential(state: PyTree, sample: PyTree,
                        reference_data: MiniBatch):
      # Approximate the potential by taking the average and scaling it to the
      # full data set size
      batch_data, batch_information = reference_data
      N = batch_information.observation_count
      n = batch_information.batch_size
      if has_state:
        batch_likelihood, new_state = likelihood(state, sample, batch_data)
      else:
        batch_likelihood = likelihood(sample, batch_data)
        new_state = None
      stochastic_potential = - N / n * batch_likelihood
      return stochastic_potential, new_state
  elif strategy == 'map':
    def batch_potential(state: PyTree, sample: PyTree, reference_data: MiniBatch):
      # The sample stays the same, therefore, it should be added to the
      # likelihood.
      if has_state:
        marginal_likelihood = partial(likelihood, state, sample)
      else:
        marginal_likelihood = partial(likelihood, sample)
      batch_data, batch_information = reference_data
      N = batch_information.observation_count
      n = batch_information.batch_size
      def single_likelihood_evaluation(cumsum, observation):
        if has_state:
          likelihood_value, state = marginal_likelihood(observation)
        else:
          likelihood_value = marginal_likelihood(observation)
          state = None
        next_cumsum = cumsum + likelihood_value
        return next_cumsum, state

      # Approximate the potential by taking the average and scaling it to the
      # full data set size
      startsum = jnp.array([0.0])
      cumsum, new_states = lax.scan(
        single_likelihood_evaluation,
        startsum,
        batch_data)
      stochastic_potential = - N / n * cumsum
      if has_state:
        new_state = tree_util.tree_map(lambda ary, org: jnp.reshape(jnp.take(ary, 0, axis=0), org.shape), new_states, state)
      else:
        new_state = None
      return stochastic_potential, new_state
  elif strategy in ('vmap', 'pmap'):
    if strategy == 'pmap':
      if has_state:
        batched_likelihood = pmap(likelihood,
                                  in_axes=(None, None, 0))
      else:
        batched_likelihood = pmap(likelihood,
                                  in_axes=(None, 0))
    if strategy == 'vmap':
      if has_state:
        batched_likelihood = vmap(likelihood,
                                  in_axes=(None, None, 0))
      else:
        batched_likelihood = vmap(likelihood,
                                  in_axes=(None, 0))
    def batch_potential(state: PyTree, sample: PyTree, reference_data: MiniBatch):
      # Approximate the potential by taking the average and scaling it to the
      # full data set size
      batch_data, batch_information = reference_data
      N = batch_information.observation_count
      if has_state:
        batch_likelihoods, new_states = batched_likelihood(state, sample, batch_data)
        new_state = tree_util.tree_map(lambda ary, org: jnp.reshape(jnp.take(ary, 0, axis=0), org.shape), new_states, state)
      else:
        batch_likelihoods = batched_likelihood(sample, batch_data)
        new_state = None
      stochastic_potential = - N * jnp.mean(batch_likelihoods, axis=0)
      return stochastic_potential, new_state
  else:
    raise NotImplementedError(f"Strategy {strategy} is unknown")

  def potential_function(sample: PyTree,
                         reference_data: MiniBatch,
                         state: PyTree = None) -> Tuple[Array, PyTree]:
    # Never differentiate w. r. t. reference data
    reference_data = lax.stop_gradient(reference_data)

    # Evaluate the likelihood and model for each reference data sample
    # likelihood_value = batched_likelihood_and_model(sample, reference_data)
    # It is also possible to combine the prior and the likelihood into a single
    # callable.

    likelihood_value, new_state = batch_potential(state, sample, reference_data)

    # The prior has to be evaluated only once, therefore the extra call
    prior_value = prior(sample)

    return jnp.squeeze(likelihood_value - prior_value), new_state

  return potential_function

# Todo: Implement gradient over potential evaluation

# Todo: Implement evaluation via map
# Todo: Implement batched evaluation via vmap
# Todo: Implement parallelized batched evaluation via pmap

def full_potential(prior: Callable[[PyTree], Array],
                   likelihood: Callable[[PyTree, PyTree], Array],
                   full_data_map: Callable,
                   strategy: AnyStr = "map",
                   ) -> FullPotential:
  """Transforms a pdf to compute the full potential over all reference data.

  Args:
    prior: Probability density function which is evaluated for a single
      sample.
    likelihood: Probability density function. If ``has_state = True``, then the
      first argument is the model state, otherwise the arguments are ``sample,
      reference_data``.
    full_data_map: Maps the likelihood over the complete reference data.
      Returned from :func:`jax_sgmc.data.full_reference_data`
    strategy: Determines how to evaluate the model function with respect for
      sample:

      - ``'map'`` sequential evaluation
      - ``'vmap'`` parallel evaluation via vectorization
      - ``'pmap'`` parallel evaluation on multiple devices

    has_state: If an additional state is provided for the model evaluation
    is_batched: If likelihood expects a batch of observations instead of a
      single observation. If the likelihood is batched, choosing the strategy
      has no influence on the computation.

  Returns:
    Returns a function which evaluates the stochastic potential for a mini-batch
    of data. The first argument are the latent variables and the second is the
    mini-batch.

  """

  # body_potential = minibatch_potential()


  # return potential_function

# Todo: Implement helper function to build the likelihood from the model and a
#       likelihood distribution.
