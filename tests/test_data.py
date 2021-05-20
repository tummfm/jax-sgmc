"""Test the data loaders. """

import sys

sys.path.append('..')

from jax_sgmc import data

import numpy as onp

import pytest

class TestNumpyLoader:

  def setup_numpy_loader(self, batch_size, **kwargs):
    data_loader = data.NumpyDataLoader(batch_size, **kwargs)
    chain_id = data_loader.register_random_pipeline()
    return lambda :data_loader.random_batches(chain_id)

  def test_batch_size(self):
    """Generate an array and matrix with different dimensions and test shape of
    the result."""

    batch_size = 10

    obs = [onp.ones((4,)),
           onp.ones((40,)),
           onp.ones((50, 60)),
           onp.ones((60, 70, 80))]

    par = [onp.ones((4,)),
           onp.ones((40,)),
           onp.ones((50,50)),
           onp.ones((60, 50, 50))]

    for ob, pa in zip(obs, par):

      random_batch_fn = self.setup_numpy_loader(batch_size, obs=ob, par=pa)
      batch = random_batch_fn()

      print(batch)

      test_obs_batch = batch["obs"][0]
      test_par_batch = batch["par"][0]

      # Count of samples is correct
      assert test_obs_batch.shape[0] == batch_size
      assert test_par_batch.shape[0] == batch_size

      # Shape of samples is correct
      assert test_obs_batch.shape[1:] == ob.shape[1:]
      assert test_par_batch.shape[1:] == pa.shape[1:]


  def test_relation(self):
    """Test if sample and observations are corresponding"""

    obs = onp.arange(10)
    par = onp.arange(10)

    random_batch_fn = self.setup_numpy_loader(batch_size=2, obs=obs, par=par)

    for i in range(100):
      batch = random_batch_fn()
      obs_batch, par_batch = batch["obs"], batch["par"]

      assert onp.sum(obs_batch) == onp.sum(par_batch)
