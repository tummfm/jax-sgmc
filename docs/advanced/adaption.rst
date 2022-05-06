Extend Adapted Quantities
=========================

Extension of Adaption Strategies
_________________________________

Each adaption strategy is expected to return three functions

::

  @adaption(quantity=SomeQuantity)
  def some_adaption(minibatch_potential: Callable = None):
    ...
    return init_adaption, update_adaption, get_adaption

The decorator :func:`adaption` wraps all three functions to flatten pytrees to
1D-arrays and unflatten the results of :func:`get_adaption`.

The rule is that all arguments which are passed by position are expected
to have the same shape as the sample pytree and are flattened to 1D-arrays.
Arguments which should not be raveled should be passed by keyword.

1. :func:`init_adaption`

  This function initializes the state of the adaption and the ravel- and unravel
  functions. Therefore, it must accept at least one positional argument with
  the shape of the sample pytree.

  ::

    ...
    def init_adaption(sample, momentum, parameter = 0.5):
      ...

  In the above example, the sample and the momentum are 1D-arrays with size
  equal to the latent variable count. Parameter is a scalar and will not be
  raveled.

2. :func:`update_adaption`

  This function updates the state of the adaption. It must accept at least one
  positional argument, the state, even if the adaption is stateless.

  ::

    ...
    # This is a stateless adaption
    def update_adaption(state, *args, **kwargs):
      del state, args, kwargs
      return None

  If the factory function of the adaption strategy is called with a potential
  function as keyword argument (`minibatch_potential = some_fun`), then
  :func:`update_adaption` is additionally called with the keyword arguments
  `flat_potential` and `mini_batch`. `flat_potential` is a wrapped version of
  the original potential function and can be called with the raveled sample.

3. :func:`get_adaption`

  This function calculates the desired quantity. Its argument-signature equals
  :func:`update_adaption`. It should return a 1D tuple of values in the right
  order, such that the quantity of the type ``NamedTuple`` can be created by
  providing positional arguments. For example, if the quantity has
  the fields `q = namedtuple('q', ['a', 'b', 'c'])`, the get function should
  look like

  ::

    ...
    def get_adaption(state, *args, **kwargs):
      ...
      return a, b, c

  The returned arrays can have dimension 1 or 2.


Extension of Quantities
_________________________

The introduction of quantities simplifies the implementation into an integrator
or solver.

For example, adapting a manifold :math:`G` for SGLD requires the calculation of
:math:`G^{-1},\ G^{-\\frac{1}{2}},\ \\text{and}\ \\Gamma`. If
:func:`get_adaption` returns all three quantities in the order

::

  @adaption(quantity=Manifold)
  def some_adaption():
    ...
    def get_adaption(state, ...):
      ...
      return g_inv, g_inv_sqrt, gamma

the manifold should be defined as following, where the correct order of
filed names is important:

::

  class Manifold(NamedTuple):
    g_inv: PyTree
    g_inv_sqrt: PyTree
    gamma: PyTree

The new :func:`get_adaption` does only return a single value of type
:class:`Manifold`.

::

  init_adaption, update_adaption, get_adaption = some_adaption()
  ...
  G = get_adaption(state, ...)
