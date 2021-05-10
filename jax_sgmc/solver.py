"""Solvers for Stochastic Gradient Bayesian Inference."""

from typing import Callable, Any, Optional

PyTree = Any


# Todo: Implement kernel
# Todo: Implement chain runner
# Todo: Implement saving module


def sgld(*args, **kwargs) -> Callable[[Any, Any], PyTree]:
  """Initializes the standard SGLD - sampler. """

  assert False, "Currently not implemented"

  # Todo: Initilize the submodules:
  # - Chain runner , e. g. simple chain or parallel chains (vmap)
  # - Kernel, simple SGLD kernel (with preconditioning)
  # - Pruning
  # - Saving
  # - Step Size schedule => Could be input

  # Todo: Run the function
