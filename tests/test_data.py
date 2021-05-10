"""Test the data loaders. """

import sys

sys.path.append('..')

from jax_sgmc import data

import numpy as onp

import pytest

class TestData:
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
      data_loader = data.PreloadReferenceData(ob,
                                              parameters=pa,
                                              batch_size=batch_size)

      test_obs_batch, test_par_batch = data_loader.get_random_batch()

      # Count of samples is correct
      assert test_obs_batch.shape[0] == batch_size
      assert test_par_batch.shape[0] == batch_size

      # Shape of samples is correct
      assert test_obs_batch.shape[1:] == ob.shape[1:]
      assert test_par_batch.shape[1:] == pa.shape[1:]

  def test_different_sample_size(self):
    """Test that parameters and observations must have same sample count."""

    obs = onp.ones((14, 10))
    par = onp.ones((15, 10))

    with pytest.raises(AssertionError):
      data_loader = data.PreloadReferenceData(obs, par)

  def test_relation(self):
    """Test if sample and observations are corresponding"""

    obs = onp.arange(10)
    par = onp.arange(10)

    data_loader = data.PreloadReferenceData(obs, par, batch_size=2)

    for i in range(100):
      obs_batch, par_batch = data_loader.get_random_batch()

      assert onp.sum(obs_batch) == onp.sum(par_batch)

  def test_no_par(self):

    obs = onp.ones((14, 15))

    data_loader = data.PreloadReferenceData(obs)

    obs_batch, par_batch = data_loader.get_random_batch()

    assert par_batch == None