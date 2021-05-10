"""Data input and output."""

from typing import Tuple

import jax.numpy as jnp

from jax import random

import numpy as onp

from jax_sgmc.util import Array

# Todo: Implement class to handle reference data without exposing dataloader
# Todo: Implement basic data loading
# Todo: Implement data loading with tensorflow

class ReferenceData:
  """Interface to the reference data.

  Reference data must be invariant under pmap and vmap.



  """

  def get_random_batch(self):
    """Get a random batch."""

    # The key makes it possible to cache the data, but it would be better to
    # eliminate the requirement of passing a key. This could be done by
    # changing the implementation of the batch potential evaluation

    pass

  def __iter__(self):
    """Iterating over all batches."""



class PreloadReferenceData(ReferenceData):
  """Reference data load into memory."""

  def __init__(self, observations, parameters=None, batch_size=1):
    self.key = random.PRNGKey(0)
    self.batch_size = batch_size

    # Todo: Check, that the batch shape is not bigger than the sample count

    # We put the data on the default device by transforming it to a default
    # numpy array

    self.observations = onp.array(observations)

    # If parameters are passed, for each observation sample a parameter sample
    # must exists

    if parameters is None:
      self.parameters = None
    else:
      self.parameters =  onp.array(parameters)

      assert self.parameters.shape[0] == self.observations.shape[0]


  def get_random_batch(self) -> Tuple[Array, Array]:

    # Each batch is assembled by draw from all samples with equal probability

    split, self.key = random.split(self.key)

    sample_count = self.observations.shape[0]
    sample_selection = random.choice(split,
                                     jnp.arange(sample_count),
                                     shape=(self.batch_size,))
    sample_selection = tuple(onp.array(sample_selection))

    # We need to distinguish between one and more dimensional arrays for
    # indexing

    if self.observations.ndim == 1:
      observations_random_batch = jnp.array(
        self.observations[sample_selection,])
    else:
      observations_random_batch = jnp.array(
        self.observations[sample_selection,::])

    if self.parameters is None:
      parameters_random_batch = None
    else:
      if self.parameters.ndim == 1:
        parameters_random_batch = jnp.array(self.parameters[sample_selection,])
      else:
        parameters_random_batch = jnp.array(self.parameters[sample_selection,::])

    return observations_random_batch, parameters_random_batch


def checkpoint(*args, **kwargs):
  """Saves complete state"""
  pass