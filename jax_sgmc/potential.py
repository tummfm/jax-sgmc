"""Utility to evaluate stochastic or real potential.

Stochastic gradient monte carlo requires to evaluate the potential and the model
for a multiple of observations or all observations. However, the likelihood and
model function only accept a singe observation and parameter set. Therefore,
this module maps the evaluation over the mini-batch or even all observations by
making use of jaxs tools ``map``, ``vmap`` and ``pmap``.

"""

# Todo: Usage example

from functools import partial

from typing import Callable, Any, AnyStr

from jax import vmap, pmap
from jax import lax

import jax.numpy as jnp

from jax_sgmc import util
from jax_sgmc.data import full_data_state, MiniBatch

# Here we define special types

PyTree = Any
Array = util.Array
Potential = Callable[[PyTree, full_data_state], Array]

Likelihood = Callable[[PyTree, MiniBatch], Array]
Prior = Callable[[PyTree], Array]

StochasticPotential = Callable[[PyTree, MiniBatch], Array]
FullPotential = Callable[[PyTree, full_data_state], Array]

# Todo: Possibly support soft-vmap (numpyro)
# Todo: Implement evaluation via pmap

def minibatch_potential(prior: Prior,
                        likelihood: Likelihood,
                        strategy: AnyStr = "map"
                        ) -> StochasticPotential:
  """Initializes the potential function for a minibatch of data.

  Args:
    prior: Probability density function which is evaluated for a single
      sample.
    likelihood: Probability density function which is evaluated for a single
      first argument but multiple second arguments. The likelihood includes the
      model evaluation.
    strategy: Determines hwo to evaluate the model function with respect for
      sample:

      - ``'map'`` sequential evaluation
      - ``'vmap'`` parallel evaluation via vectorization
      - ``'pmap'`` parallel evaluation on multiple devices

  Returns:
    Returns a function which evaluates the stochastic potential for a mini-batch
    of data. The first argument are the latent variables and the second is the
    mini-batch.
  """

  # The final function to evaluate the potential including likelihood and prio

  # Todo: Possibly distinguish between constant/none parameters and pairs of
  #       reference data and parameters

  if strategy == 'map':
    def batch_potential(sample, reference_data: MiniBatch):
      # The sample stays the same, therefore, it should be added to the
      # likelihood.
      marginal_likelihood = partial(likelihood, sample)
      batch_data, batch_information = reference_data
      N = batch_information.observation_count
      n = batch_information.batch_size
      def single_likelihood_evaluation(cumsum, observation):
        likelihood_value = marginal_likelihood(observation)
        next_cumsum = cumsum + likelihood_value
        return next_cumsum, None

      # Approximate the potential by taking the average and scaling it to the
      # full data set size
      startsum = jnp.array([0.0])
      cumsum, _ = lax.scan(single_likelihood_evaluation,
                           startsum,
                           batch_data)
      stochastic_potential = - N / n * cumsum
      return stochastic_potential
  elif strategy in ('vmap', 'pmap'):
    if strategy == 'pmap':
      batched_likelihood = pmap(likelihood,
                                in_axes=(None, 0))
    if strategy == 'vmap':
      batched_likelihood = vmap(likelihood,
                                in_axes=(None, 0))
    def batch_potential(sample, reference_data: MiniBatch):
      # Approximate the potential by taking the average and scaling it to the
      # full data set size
      batch_data, batch_information = reference_data
      N = batch_information.observation_count
      batch_likelihoods = batched_likelihood(sample, batch_data)
      stochastic_potential = - N * jnp.mean(batch_likelihoods, axis=0)
      return stochastic_potential
  else:
    raise NotImplementedError(f"Strategy {strategy} is unknown")

  def potential_function(sample: PyTree,
                         reference_data: MiniBatch
                         ) -> Array:
    # Never differentiate w. r. t. reference data
    reference_data = lax.stop_gradient(reference_data)

    # Evaluate the likelihood and model for each reference data sample
    # likelihood_value = batched_likelihood_and_model(sample, reference_data)
    # It is also possible to combine the prior and the likelihood into a single
    # callable.

    likelihood_value = batch_potential(sample, reference_data)

    # The prior has to be evaluated only once, therefore the extra call
    prior_value = prior(sample)

    return jnp.squeeze(likelihood_value - prior_value)

  return potential_function

# Todo: Implement gradient over potential evaluation

# Todo: Implement evaluation via map
# Todo: Implement batched evaluation via vmap
# Todo: Implement parallelized batched evaluation via pmap

def full_potential(prior: Callable[[PyTree], Array],
                   likelihood: Callable[[PyTree, PyTree], Array],
                   strategy: AnyStr = "map"
                   ) -> Callable[[PyTree, full_data_state], Array]:
  """Transforms a pdf to compute the full potential over all reference data.

  Args:
      prior: Probability density function which is evaluated for a single
          sample.
      likelihood: Probability density function which is evaluated for a single
          first argument but multiple second arguments.

  Returns:
      Returns a function which evaluates the stochastic potential for all
      reference data. The reference data is accessed by providing an instance
      of the class `ReferenceData`.

  """

  # Will be needed here as this function just loops over all minibatches
  # Todo: Implement strategy to evaluate mini-batches for full data

  # minibatch_eval = minibatch_potential()

  assert False, "Currently not implemented"

  # The final function to evaluate the potential including likelihood and prio

  # def potential_function(sample: PyTree,
  #                        refernce_data: ReferenceData
  #                        ) -> Array:
  #   pass

  # return potential_function

# Todo: Implement helper function to build the likelihood from the model and a
#       likelihood distribution.
