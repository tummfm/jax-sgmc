from collections import  namedtuple

import pytest

import os
from jax.lib import xla_bridge

from jax import test_util

# Setup multi-device environment
@pytest.fixture(scope='session', autouse=True)
def pmap_setup():
  # Setup
  prev_xla_flags = os.getenv("XLA_FLAGS")
  flags_str = prev_xla_flags or ""
  # Don't override user-specified device count, or other XLA flags.
  if "xla_force_host_platform_device_count" not in flags_str:
    os.environ["XLA_FLAGS"] = (flags_str +
                               " --xla_force_host_platform_device_count=12")
  # Clear any cached backends so new CPU backend will pick up the env var.
  xla_bridge.get_backend.cache_clear()

  # Run
  yield

  # Reset to previous configuration in case other test modules will be run.
  if prev_xla_flags is None:
    del os.environ["XLA_FLAGS"]
  else:
    os.environ["XLA_FLAGS"] = prev_xla_flags
  xla_bridge.get_backend.cache_clear()


