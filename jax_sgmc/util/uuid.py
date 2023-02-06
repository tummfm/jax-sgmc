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

import uuid

import numpy as onp

import jax.numpy as jnp
from jax import tree_util

from jax import Array

@tree_util.register_pytree_node_class
class JaxUUID:

  def __init__(self, ints: Array = None):
    if ints is None:
      uuid_int = uuid.uuid4().int
      ints = [(uuid_int >> bits) & 0xFFFFFFFF for bits in range(0, 128, 32)]
      ints = onp.array(ints, dtype=jnp.int32)

    self._uuid_int = ints

  @property
  def as_uuid(self):
    # Rearrange 4 int32 to uuid
    shifted_ints = [int(int(sint) << bits)
                    for sint, bits
                    in zip(self._uuid_int, range(0, 128, 32))]
    int128 = sum(shifted_ints)
    # Ensure that the hex number has exactly 128 bits and is non-negative
    hex128 = hex(int128).replace('0x', '').replace('-','').zfill(32)
    return uuid.UUID(hex128)

  def __repr__(self):
    return str(self.as_uuid)

  @property
  def as_int32s(self):
    return self._uuid_int

  def tree_flatten(self):
    # Wrapping the ints in a tuple ensures that they remain a single array of
    # length 4.
    children = (self._uuid_int,)
    return (children, None)

  @classmethod
  def tree_unflatten(cls, _, children):
    ints, = children
    return cls(ints=ints)
