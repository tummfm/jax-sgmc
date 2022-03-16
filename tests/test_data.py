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

import numpy as onp

import jax.numpy as jnp
from jax import ShapeDtypeStruct

from jax_sgmc.data.numpy_loader import NumpyDataLoader

import pytest

@pytest.fixture
def dataset():
  obs_count = 13

  data = {
    "ordered_indices": jnp.arange(obs_count, dtype=jnp.int_),
    "all_ones": jnp.ones(obs_count, dtype=jnp.bool_),
    "shape_5": jnp.ones((obs_count, 5), dtype=jnp.float_),
    "shape_3_5": jnp.ones((obs_count, 3, 5), dtype=jnp.int_)
  }

  data_format = {key: ShapeDtypeStruct(dtype=value.dtype, shape=value.shape)
                 for key, value in data.items()}

  return data, data_format


class TestNumpyDeviceDataLoader:

  @pytest.fixture
  def data_loader(self, dataset):
    data, data_format = dataset
    return NumpyDataLoader(**data), data_format


class TestNumpyHostDataLoader:
  pass


class TestTensorflowDataLoader:
  pass


class TestHostCallback:
  pass


class TestRandomDataAccess:
  pass


class TestFullDataAccess:
  pass
