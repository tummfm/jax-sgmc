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

.. doctest::

  >>> from functools import partial
  >>> import jax.numpy as jnp
  >>> import jax.scipy as jscp
  >>> from jax import random, vmap
  >>> from jax_sgmc import data, potential

  >>> mean = random.normal(random.PRNGKey(0), shape=(100, 5))
  >>> data_loader = data.NumpyDataLoader(mean=mean)
  >>>
  >>> test_sample = {'mean': jnp.zeros(5), 'std': jnp.ones(1)}


Stochastic Potential
---------------------

The stochastic potential is an estimation of the true potential. It is
calculated over a small dataset and rescaled to the full dataset.

  >>> batch_init, batch_get = data.random_reference_data(data_loader,
  ...                                                    cached_batches_count=50,
  ...                                                    mb_size=5)
  >>> random_data_state = batch_init()

Unbatched Likelihood
_____________________

The likelihood can be written for a single observation. The
:mod:`jax_sgmc.potential` module then evaluates the likelihood for a batch of
reference data sequentially via ``map`` or parallel via ``vmap`` or ``pmap``.

  >>> def likelihood(sample, observation):
  ...   likelihoods = jscp.stats.norm.logpdf(observation['mean'],
  ...                                        loc=sample['mean'],
  ...                                        scale=sample['std'])
  ...   return jnp.sum(likelihoods)
  >>> prior = lambda unused_sample: 0.0
  >>>
  >>> stochastic_potential_fn = potential.minibatch_potential(prior,
  ...                                                         likelihood,
  ...                                                         strategy='map')
  >>> new_random_data_state, random_batch = batch_get(random_data_state, information=True)
  >>> potential_eval, unused_state = stochastic_potential_fn(test_sample, random_batch)
  >>>
  >>> print(potential_eval)
  883.183

Batched Likelihood
___________________

Some models already accept a batch of reference data. In this case, the
potential function can be constructed by setting ``is_batched = True``. In this
case, it is expected that the returned likelihoods are a vectore with shape
``(N,)``, where N is the batch-size.


  >>> @partial(vmap, in_axes=(None, 0))
  ... def batched_likelihood(sample, observation):
  ...   likelihoods = jscp.stats.norm.logpdf(observation['mean'],
  ...                                        loc=sample['mean'],
  ...                                        scale=sample['std'])
  ...   # Only valid samples contribute to the likelihood
  ...   return jnp.sum(likelihoods)
  >>>
  >>> stochastic_potential_fn = potential.minibatch_potential(prior,
  ...                                                         batched_likelihood,
  ...                                                         is_batched=True,
  ...                                                         strategy='map')
  >>>
  >>> new_random_data_state, random_batch = batch_get(random_data_state, information=True)
  >>> potential_eval, unused_state = stochastic_potential_fn(test_sample, random_batch)
  >>>
  >>> print(potential_eval)
  883.183
  >>>
  >>> _, (likelihoods, _) = stochastic_potential_fn(test_sample,
  ...                                               random_batch,
  ...                                               likelihoods=True)
  >>>
  >>> print(jnp.var(likelihoods))
  7.45549


Full Potential
---------------

In combination with the :mod:`jax_sgmc.data` it is possible to calculate the
true potential over the full dataset.
If we specify a batch size of 3, then the liklihood will be sequentially
calculated over batches with the size 3.


  >>> init_fun, fmap_fun = data.full_reference_data(data_loader,
  ...                                               cached_batches_count=50,
  ...                                               mb_size=3)
  >>> data_state = init_fun()

Unbatched Likelihood
_____________________

Here, the likelihood written for a single observation can be re-used.

  >>> potential_fn = potential.full_potential(prior, likelihood, strategy='vmap')
  >>>
  >>> potential_eval, (data_state, unused_state) = potential_fn(
  ...   test_sample, data_state, fmap_fun)
  >>>
  >>> print(potential_eval)
  707.4376

Bached Likelihood
__________________

The batched likelihood can also be used to calculate the full potential.

  >>> prior = lambda unused_sample: 0.0
  >>>
  >>> potential_fn = potential.full_potential(prior, batched_likelihood, is_batched=True)
  >>>
  >>> potential_eval, (data_state, unused_state) = potential_fn(
  ...   test_sample, data_state, fmap_fun)
  >>>
  >>> print(potential_eval)
  707.4376


Likelihoods with States
------------------------

By setting the argument ``has_state = True``, the likelihood accepts a state
as first positional argument.

  >>> def statefull_likelihood(state, sample, observation):
  ...   n, mean = state
  ...   n += 1
  ...   new_mean = (n-1)/n * mean + 1/n * observation['mean']
  ...
  ...   likelihoods = jscp.stats.norm.logpdf((observation['mean'] - new_mean),
  ...                                        loc=(sample['mean'] - new_mean),
  ...                                        scale=sample['std'])
  ...   return jnp.sum(likelihoods), (n, new_mean)
  >>>
  >>> potential_fn = potential.minibatch_potential(prior,
  ...                                              statefull_likelihood,
  ...                                              has_state=True)
  >>>
  >>> potential_eval, new_state = potential_fn(test_sample,
  ...                                          random_batch,
  ...                                          state=(jnp.array(2), jnp.ones(5)))
  >>>
  >>> print(potential_eval)
  883.183
  >>> print(new_state)
  (DeviceArray(3, dtype=int32), DeviceArray([0.79154414, 0.9063752 , 0.52024883, 0.3007263 , 0.10383289],            dtype=float32))

"""

from functools import partial

from typing import Callable, Any, AnyStr, Optional, Tuple, Union, Protocol

from jax import vmap, pmap, lax, tree_util, named_call

import jax.numpy as jnp

from jax_sgmc import util
from jax_sgmc.data import CacheState, MiniBatch

# Here we define special types

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
                          Tuple[Array, Tuple[Array, PyTree]]]: ...

class FullPotential(Protocol):
  def __call__(self,
               sample: PyTree,
               data_state: CacheState,
               full_data_map_fn: Callable,
               state: PyTree = None
               ) -> Tuple[Array, Tuple[CacheState, PyTree]]: ...


# Todo: Possibly support soft-vmap (numpyro)

def minibatch_potential(prior: Prior,
                        likelihood: Callable,
                        strategy: AnyStr = "map",
                        has_state: bool = False,
                        is_batched: bool = False) -> StochasticPotential:
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
        jnp.squeeze(batch_likelihood - prior_value),
        (observation_likelihoods, new_state))
    else:
      return jnp.squeeze(batch_likelihood - prior_value), new_state

  return potential_function


def full_potential(prior: Callable[[PyTree], Array],
                   likelihood: Callable[[PyTree, PyTree], Array],
                   strategy: AnyStr = "map",
                   has_state = False,
                   is_batched = False
                   ) -> FullPotential:
  """Transforms a pdf to compute the full potential over all reference data.

  Args:
    prior: Probability density function which is evaluated for a single
      sample.
    likelihood: Probability density function. If ``has_state = True``, then the
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
    data_state, (results, new_state) = full_data_map_fn(
      body_fn, data_state, state, masking=True, information=True)

    # The prior needs just a single evaluation
    prior_eval = prior(sample)

    return jnp.squeeze(jnp.sum(results) - prior_eval), (data_state, new_state)

  return sum_batched_evaluations

# Todo: Implement helper function to build the likelihood from the model and a
#       likelihood distribution.
