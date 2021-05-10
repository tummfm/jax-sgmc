"""Defines prior distributions and likelihoods"""


import jax

from jax.tree_util import register_pytree_node_class


#@register_pytree_node_class
class DistributionNode:
  """Base class representing a distribution"""


class Distribution:
  """Base class for distributions which can be evaluated over pytrees."""

  def __init__(self):

    assert False, "Currently not implemented"


class Normal(DistributionNode):
  pass

class Uniform(DistributionNode):
  pass