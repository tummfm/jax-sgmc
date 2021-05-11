"""Utility to evaluate stochastic or real potential.

Stochastic gradient monte carlo requires to evaluate the potential and the model
for a multiple of observations or all observations. However, the likelihood and model
funtion only accept a singe observation and parameter set. Therefore, this
module maps the evaluation over the mini-batch or even all observations by
making use of jaxs tools ``map``, ``vmap`` and ``pmap``.

"""

# Todo: Usage example

from functools import partial

from typing import Callable, Any, AnyStr

from jax import vmap, grad
from jax.lax import scan

import jax.numpy as jnp

from jax_sgmc import util
from jax_sgmc.data import ReferenceData, mini_batch

# Here we define special types

PyTree = Any
Array = util.Array
Potential = Callable[[PyTree, ReferenceData], Array]
Likelihood = Callable[[PyTree, Array, Array], Array]
Prior = Callable[[PyTree], Array]
Model = Callable[[PyTree, Array], PyTree]

# Todo: Implement evaluation via map
# Todo: Implement evaluation via vmap
# Todo: Implement evaluation via pmap

def minibatch_potential(prior: Prior,
                        likelihood: Likelihood,
                        strategy: AnyStr = "map"
                        ) -> Callable[[PyTree, mini_batch], Array]:
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
      - ``'pmap'`` parallel evaluation

  Notes:
    The sample is a dict of the form ``dict(model=model_latent_variables,
    likelihood_latent_varaiables)``.

  Returns:
    Returns a function which evaluates the stochastic potential for a mini-batch
    of data. The first argument are the latent variables and the second is the
    minibatch.

  """

  # The final function to evaluate the potential including likelihood and prio

  # Todo: Possibly distinguish between constant/none parameters and pairs of
  #       reference data and parameters

  if strategy == 'map':
    def batch_potential(sample, reference_data: mini_batch):
      # The sample stays the same, therefore, it should be added to the
      # likelihood.
      marginal_likelihood = partial(likelihood, sample)

      def single_likelihood_evaluation(cumsum, reference_data_pair):
        observations, parameters = reference_data_pair
        likelihood_value = marginal_likelihood(parameters, observations)
        next_cumsum = cumsum + likelihood_value
        return next_cumsum, None

      # Approximate the potential by taking the average and scaling it to the
      # full data set size
      startsum = jnp.float32(0.0)
      N = reference_data.observation_count
      n = reference_data.mini_batch[0].shape[0] # The size of the mini_batch
      cumsum, _ = scan(single_likelihood_evaluation,
                       startsum,
                       reference_data.mini_batch)
      stochastic_potential = - N / n * cumsum
      return stochastic_potential
  elif strategy == 'vmap':
    @partial(vmap, in_axes=(None, 0))
    def vmap_helper(sample, reference_data_pair):
      observations, parameters = reference_data_pair
      likelihood_value = likelihood(sample, parameters, observations)
      return likelihood_value

    def batch_potential(sample, reference_data: mini_batch):
      # Approximate the potential by taking the average and scaling it to the
      # full data set size
      N = reference_data.observation_count
      batch_likelihoods = vmap_helper(sample, reference_data.mini_batch)
      stochastic_potential = - N * jnp.mean(batch_likelihoods, axis=0)
      return jnp.float32(stochastic_potential)
  else:
    assert False, "Currently not implemented"

  def potential_function(sample: PyTree,
                         reference_data: mini_batch
                         ) -> Array:

    # Evaluate the likelihood and model for each reference data sample
    # liklihood_value = batched_likelihood_and_model(sample, reference_data)

    likelihood_value = batch_potential(sample, reference_data)

    # The prior has to be evaluated only once, therefore the extra call
    prior_value = prior(sample)

    return likelihood_value - prior_value

  return potential_function

# Todo: Implement gradient over potential evaluation

# Todo: Implement evaluation via map
# Todo: Implement batched evaluation via vmap
# Todo: Implement parallelized batched evaluation via pmap

def full_potential(prior: Callable[[PyTree], Array],
                   likelihood: Callable[[PyTree, PyTree], Array],
                   strategy: AnyStr = "map"
                   ) -> Callable[[PyTree, ReferenceData], Array]:
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

  minibatch_eval = minibatch_potential()

  assert False, "Currently not implemented"

  # The final function to evaluate the potential including likelihood and prio

  def potential_function(sample: PyTree,
                         refernce_data: ReferenceData
                         ) -> Array:
    pass

  return potential_function

def stochastic_potential_gradient(prior: Prior,
                                  likelihood: Likelihood,
                                  strategy: AnyStr = "map"
                                  ) -> Callable[[PyTree, mini_batch], Array]:
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
      - ``'pmap'`` parallel evaluation

  Notes:
    The sample is a dict of the form ``dict(model=model_latent_variables,
    likelihood_latent_varaiables)``.

    Returns:
      Returns a function which evaluates the stochastic gradient of the
      potential for a mini-batch of data. The first argument are the latent
      variables and the second is the minibatch.

    """

  # The stochastic potential gradient ist a wrapper around the potential fuction
  # to simplify the gradient computation. Therefore, the potential function must
  # be initialized first

  potential_fn = minibatch_potential(prior,
                                     likelihood,
                                     strategy)

  # Only the gradient over the sample values is important

  gradient = grad(potential_fn, argnums=0)

  return gradient

# Todo: Implement helper function to build the likelihood from the model and a
#       likelihood distribution.