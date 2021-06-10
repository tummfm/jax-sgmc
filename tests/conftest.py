from collections import  namedtuple

import pytest

from jax_sgmc import util

pmap_setup_state = namedtuple("pmap_setup_state",
                              ["host_count"])

# # Setup multi-device environment
# @pytest.fixture(scope='session')
# def pmap_setup():
#   hc = 5 # Host count
#   util.set_host_device_count(hc)
#   return pmap_setup_state(hc)
