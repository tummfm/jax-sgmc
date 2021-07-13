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
potential function can be constructed by setting ``is_batched = True``.
The mask provides information about which of the batched observations are valid.
Invalid samples only occur during the full potential evaluation, so during the
stochastic potential evaluation no mask argument is provided.

Some solver need information about the variance of the stochastic potential over
the reference batch. Therefore, this variance needs to be passed by the batched
likelihood function. However, as the variance is only used for the stochastic
potential and not for the full potential, a nan value can be returend for the
masked evaluations.

  >>> @partial(vmap, in_axes=(None, 0))
  ... def batch_eval(sample, observation):
  ...   likelihoods = jscp.stats.norm.logpdf(observation['mean'],
  ...                                        loc=sample['mean'],
  ...                                        scale=sample['std'])
  ...   # Only valid samples contribute to the likelihood
  ...   return jnp.sum(likelihoods)
  >>>
  >>> # This function calculates a single likelihood for a batch of observations.
  >>> def batched_likelihood(samle, observations, mask=None):
  ...   likelihoods = batch_eval(samle, observations)
  ...   # To ensure compatibility with the full potential evaluation.
  ...   if mask is None:
  ...     return jnp.sum(likelihoods), jnp.var(likelihoods)
  ...   else:
  ...     return jnp.sum(mask * likelihoods), jnp.nan
  >>>
  >>> new_random_data_state, random_batch = batch_get(random_data_state, information=True)
  >>> potential_eval, unused_state = stochastic_potential_fn(test_sample, random_batch)
  >>>
  >>> print(potential_eval)
  883.183
  >>>
  >>> _, (variance, _) = stochastic_potential_fn(test_sample,
  ...                                            random_batch,
  ...                                            variance=True)
  >>>
  >>> print(variance)
  7.4554925


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

  >>> potential_fn = potential.full_potential(prior, likelihood, fmap_fun, strategy='vmap')
  >>>
  >>> potential_eval, (data_state, unused_state) = potential_fn(test_sample, data_state)
  >>>
  >>> print(potential_eval)
  707.4376

Bached Likelihood
__________________

In the case that the likelihood accepts a batch of observations, using the
full dataset mapping requires implementing a mask. The mask provides
information whether an observation is valid or just a filler value to ensure
equal batch shapes.
Because we implemented the mask argument as kwarg in the batched likelihood for
the stochastic potential evaluation, we can easily re-use it here.

  >>> prior = lambda unused_sample: 0.0
  >>>
  >>> potential_fn = potential.full_potential(prior, batched_likelihood, fmap_fun, is_batched=True)
  >>>
  >>> potential_eval, (data_state, unused_state) = potential_fn(test_sample, data_state)
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
  >>> potential_fn = potential.minibatch_potential(prior, statefull_likelihood, has_state=True)
  >>>
  >>> potential_eval, new_state = potential_fn(test_sample, random_batch, state=(jnp.array(2), jnp.ones(5)))
  >>>
  >>> print(potential_eval)
  883.183
  >>> print(new_state)
  (DeviceArray(3, dtype=int32), DeviceArray([0.79154414, 0.9063752 , 0.52024883, 0.3007263 , 0.10383289],            dtype=float32))

"""

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
                        likelihood: Callable,
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

  # The final function to evaluate the potential including likelihood and prio
  if is_batched:
    # The likelihood is already provided for a batch of data
    def batch_potential(state: PyTree,
                        sample: PyTree,
                        reference_data: MiniBatch,
                        mask: Array):
      # Approximate the potential by taking the average and scaling it to the
      # full data set size
      batch_data, batch_information = reference_data
      N = batch_information.observation_count
      n = batch_information.batch_size
      if mask is None:
        if has_state:
          batch_likelihood, batch_variance, new_state = likelihood(
            state, sample, batch_data)
        else:
          batch_likelihood, batch_variance = likelihood(
            sample, batch_data)
          new_state = None
      else:
        if has_state:
          batch_likelihood, batch_variance, new_state = likelihood(
            state, sample, batch_data, mask)
        else:
          batch_likelihood, batch_variance = likelihood(
            sample, batch_data, mask)
          new_state = None
      stochastic_potential = - N / n * batch_likelihood
      return stochastic_potential, batch_variance, new_state
  elif strategy == 'map':
    def batch_potential(state: PyTree,
                        sample: PyTree,
                        reference_data: MiniBatch,
                        mask: Array):
      # The sample stays the same, therefore, it should be added to the
      # likelihood.
      if has_state:
        marginal_likelihood = partial(likelihood, state, sample)
      else:
        marginal_likelihood = partial(likelihood, sample)
      batch_data, batch_information = reference_data
      N = batch_information.observation_count
      n = batch_information.batch_size
      # The mask is only necessary for the full potential evaluation
      if mask is None:
        def single_likelihood_evaluation(cumsum, observation):
          if has_state:
            likelihood_value, state = marginal_likelihood(observation)
          else:
            likelihood_value = marginal_likelihood(observation)
            state = None
          next_cumsum = cumsum + likelihood_value
          return next_cumsum, (likelihood_value, state)
      else:
        def single_likelihood_evaluation(cumsum, it):
          observation, mask = it
          if has_state:
            likelihood_value, state = marginal_likelihood(observation)
          else:
            likelihood_value = marginal_likelihood(observation)
            state = None
          next_cumsum = cumsum + likelihood_value * mask
          return next_cumsum, (likelihood_value * mask, state)

      # Approximate the potential by taking the average and scaling it to the
      # full data set size
      startsum = jnp.array([0.0])
      if mask is None:
        cumsum, (likelihoods, new_states) = lax.scan(
          single_likelihood_evaluation,
          startsum,
          batch_data)
        batch_variance = jnp.var(likelihoods)
      else:
        cumsum, (likelihoods, new_states) = lax.scan(
          single_likelihood_evaluation,
          startsum,
          (batch_data, mask))
        # Todo: Variance for masked arrays
        batch_variance = jnp.nan
      stochastic_potential = - N / n * cumsum
      if has_state:
        new_state = tree_util.tree_map(
          lambda ary, org: jnp.reshape(jnp.take(ary, 0, axis=0), org.shape),
          new_states,
          state)
      else:
        new_state = None
      return stochastic_potential, batch_variance, new_state
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
    def batch_potential(state: PyTree,
                        sample: PyTree,
                        reference_data: MiniBatch,
                        mask: Array):
      # Approximate the potential by taking the average and scaling it to the
      # full data set size
      batch_data, batch_information = reference_data
      N = batch_information.observation_count
      n = batch_information.batch_size
      if has_state:
        batch_likelihoods, new_states = batched_likelihood(state, sample, batch_data)
        new_state = tree_util.tree_map(
          lambda ary, org: jnp.reshape(jnp.take(ary, 0, axis=0), org.shape),
          new_states,
          state)
      else:
        batch_likelihoods = batched_likelihood(sample, batch_data)
        new_state = None
      # The mask is only necessary for the full potential evaluation
      if mask is None:
        stochastic_potential = - N * jnp.mean(batch_likelihoods, axis=0)
        stochastic_variance = jnp.var(batch_likelihoods)
      else:
        stochastic_potential = - N / n * jnp.sum(
          jnp.multiply(batch_likelihoods, mask), axis=0)
        # Todo: Variance for masked arrays
        stochastic_variance = jnp.nan
      return stochastic_potential, stochastic_variance, new_state
  else:
    raise NotImplementedError(f"Strategy {strategy} is unknown")

  def potential_function(sample: PyTree,
                         reference_data: MiniBatch,
                         state: PyTree = None,
                         mask: Array = None,
                         variance = False) -> Tuple[Array, PyTree]:
    # Never differentiate w. r. t. reference data
    reference_data = lax.stop_gradient(reference_data)

    # Evaluate the likelihood and model for each reference data sample
    # likelihood_value = batched_likelihood_and_model(sample, reference_data)
    # It is also possible to combine the prior and the likelihood into a single
    # callable.

    likelihood_value, stochastic_variance, new_state = batch_potential(
      state, sample, reference_data, mask)

    # The prior has to be evaluated only once, therefore the extra call
    prior_value = prior(sample)

    if variance:
      return jnp.squeeze(likelihood_value - prior_value), (stochastic_variance, new_state)
    else:
      return jnp.squeeze(likelihood_value - prior_value), new_state

  return potential_function


def full_potential(prior: Callable[[PyTree], Array],
                   likelihood: Callable[[PyTree, PyTree], Array],
                   full_data_map: Callable,
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
      has no influence on the computation. In this case, the last argument of
      the likelihood should be an optional mask. The mask is an arrays with ones
      for valid observations and zeros for non-valid observations.

  Returns:
    Returns a function which evaluates the potential over the full dataset via
    a dataset mapping from the :mod:`jax_sgmc.data` module.

  """

  # Can use the potential evaluation strategy for a minibatch of data. The prior
  # must be evaluated independently.
  batch_potential = minibatch_potential(lambda sample: 0.0,
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

  def sum_batched_evaluations(sample: PyTree,
                              data_state: CacheState,
                              state: PyTree = None):
    body_fn = partial(batch_evaluation, sample)
    data_state, (results, new_state) = full_data_map(
      body_fn, data_state, state, masking=True, information=True)

    # The prior needs just a single evaluation
    prior_eval = prior(sample)

    return jnp.squeeze(jnp.sum(results) - prior_eval), (data_state, new_state)

  return sum_batched_evaluations

# Todo: Implement helper function to build the likelihood from the model and a
#       likelihood distribution.
