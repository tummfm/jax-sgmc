"""Evaluates the potential for a minibatch or all reference data.

"""

from typing import Callable, Any, AnyStr, Optional
from jax_sgmc import util

from jax_sgmc.data import ReferenceData

# Here we define special types

PyTree = Any
Array = util.Array
Potential = Callable[[PyTree, ReferenceData], Array]

# Todo: Implement evaluation via map
# Todo: Implement evaluation via vmap
# Todo: Implement evaluation via pmap

def minibatch_potential_function(prior: Callable[[PyTree], Array],
                                 likelihood: Callable[[PyTree, PyTree], Array],
                                 strategy: AnyStr = "map"
                                 ) -> Callable[[PyTree, PyTree], Array]:
  """Transforms a pdf to be applied over a minibatch.

  Args:
      prior: Probability density function which is evaluated for a single
          sample.
      likelihood: Probability density function which is evaluated for a single
          first argument but multiple second arguments.

  Returns:
      Returns a function which evaluates the stochastic potential for a mini-
      batch of data given by reference data object.

  """

  assert False, "Currently not implemented"

  # The final function to evaluate the potential including likelihood and prio

  def potential_function(sample: PyTree,
                         refernce_data: PyTree
                         ) -> Array:
    pass

  return potential_function

# Todo: Implement gradient over potential evaluation

# Todo: Implement evaluation via map
# Todo: Implement batched evaluation via vmap
# Todo: Implement parallelized batched evaluation via pmap

def full_potential_function(prior: Callable[[PyTree], Array],
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

  minibatch_eval = minibatch_potential_function()

  assert False, "Currently not implemented"

  # The final function to evaluate the potential including likelihood and prio

  def potential_function(sample: PyTree,
                         refernce_data: ReferenceData
                         ) -> Array:
    pass

  return potential_function