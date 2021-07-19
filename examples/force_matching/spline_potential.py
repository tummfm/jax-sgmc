"""Modified potential from DiffTRe.

[1] https://arxiv.org/abs/2106.01138

"""

from functools import partial
from typing import AnyStr
from typing import Callable, Tuple, Dict, Any

import jax
import jax.numpy as jnp
from DiffTRe import custom_interpolate
from jax.tree_util import register_pytree_node_class
from jax_md import space, partition, util, smap
from jax_md.energy import multiplicative_isotropic_cutoff

# Types
f32 = util.f32
f64 = util.f64
Array = util.Array

PyTree = Any
Box = space.Box
DisplacementFn = space.DisplacementFn
DisplacementOrMetricFn = space.DisplacementOrMetricFn

NeighborFn = partition.NeighborFn
NeighborList = partition.NeighborList

@register_pytree_node_class
class MonotonicInterpolate:
  """Piecewise cubic, monotonic interpolation via Steffens method [1].

  The interpolation curve is monotonic within each interval such that extrema
  can only occur at grid points. Guarantees continuous first derivatives
  of the spline. Is applicable to arbitrary data; not restricted to monotonic data.
  Contributed by Paul Fuchs.
  [1] Steffen, M., “A simple method for monotonic interpolation in one dimension.”,
  <i>Astronomy and Astrophysics</i>, vol. 239, pp. 443–450, 1990.

  Attributes:
      coefficients: Coefficients for evaluating the spline
      x: x-value of socket points
      y: y-value of socket points
      extrapolation_type: String defining type of extrapolation
      extrapolation_fn: Function used for custom extrapolation

  Args:
      x : x-value of socket points -- must be strictly increasing
      y : y-value of socket points
      extrapolation : Options for extrapolation
      coefficients: Necessary for tree_unflatten, should not be set manually
      extrapolation_type: Necessary for tree_unflatten, should not be set
       manually
      extrapolation_fn: Necessary for tree_unflatten, should not be set manually

  Example:
      Given socket points ``x_vals`` and ``y_vals``, construct spline::

          spline = MonotonicInterpolate(x_vals, y_vals)

      By default, the spline extrapolates by using the cubic splines for the
      first and last spline section. There are two options for extrapolation.
      The first one is extrapolate with a repulsive exponential potential::

          spline = MonotonicInterpolate(x_vals,
                                        y_vals,
                                        extrapolation={"type": "repulsion",
                                                       "exp": 6.0,
                                                       "sigma": 0.4 })

      The second option is to provide a custom function for extrapolation. The
      function must be autodifferentiable and the left and right onset points
      ``x0`` and ``xn`` must be smaller or respectively bigger than the socket
      points::

          spline = MonotonicInterpolate(x_vals,
                                        x_vals,
                                        extrapolation={
                                          "type": "custom",
                                          "left_fn": lambda x: 0.0,
                                          "right_fn": lambda x: jnp.cos(x),
                                          "x0": 0.0,
                                          "xn": 1.0})

      The spline can then be evaluated at the points ``r`` via::

          new_points = spline(r)

  """

  def tree_flatten(self):
    children = (self.x, self.y, self.coefficients)
    aux_data = {"extrapolation_type": self.extrapolation_type,
                "extrapolation_fn": self.extrapolation_fn}
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    x, y, coefficients = children
    return cls(x, y, coefficients=coefficients, **aux_data)

  def __repr__(self):
    return "Monotonic Spline with extrapolation type {}, x_vals={}, y_vals={}"\
      .format(self.extrapolation_type, self.x, self.y)

  def __init__(self,
               x: Array,
               y: Array,
               extrapolation: Dict = None,
               coefficients: Tuple = None,
               extrapolation_type: AnyStr = None,
               extrapolation_fn: Tuple[Callable] = None):

    # Check the input values

    assert len(x) > 3, "Not enough input values for spline (len(x) > 3"
    assert len(x) == len(y), "x and y must have the same length"
    assert x.ndim == 1 and y.ndim == 1, "Input arrays must be 1D."

    if coefficients is None:

      extrapolation_type = "default"
      extrapolation_args = ()

      h = jnp.diff(x)
      k = jnp.diff(y)
      s = k / h

      if extrapolation is None:

        p0 = s[0] * (1 + h[0] / (h[0] + h[1])) - s[1] * (h[0] / (h[0] + h[1]))
        pn = s[-1] * (1 + h[-1] / (h[-1] + h[-2])) \
             - s[-2] * (h[-1] / (h[-1] + h[-2]))

        tmp0 = jnp.where(jnp.abs(p0) > 2 * jnp.abs(s[0]), 2 * s[0], p0)
        tmpn = jnp.where(jnp.abs(pn) > 2 * jnp.abs(s[-1]), 2 * s[-1], pn)

        yp0 = jnp.where(p0 * s[0] <= 0.0, 0.0, tmp0)
        ypn = jnp.where(pn * s[-1] <= 0.0, 0.0, tmpn)

      elif extrapolation.get("type", None) == "repulsion":

        exp = extrapolation.get("exp", 6.0)
        sigma = extrapolation.get("sigma", 0.4)
        epsilon0 = extrapolation.get("epsilon", 1.0)

        epsilon1 = y[-1] * jnp.power(x[-1] / sigma, exp)

        off = y[0] - epsilon0 * jnp.power(sigma / x[0], exp)

        yp0 = -epsilon0 / sigma * exp * jnp.power(sigma / x[0], exp + 1.0)
        ypn = -epsilon1 / sigma * exp * jnp.power(sigma / x[-1], exp + 1.0)

        extrapolation_type = "repulsion"
        extrapolation_args = (exp, sigma, epsilon0, epsilon1, off)

      elif extrapolation.get("type", None) == "custom":

        x0 = extrapolation.get("x0", 0.0)
        xn = extrapolation.get("xn", 1.0)

        left_fn = extrapolation.get("left_fn", lambda x: 0.0)
        right_fn = extrapolation.get("right_fn", lambda x: 0.0)

        # We need to know the y_value at the onset points

        y0 = left_fn(x0)
        yn = right_fn(xn)

        # We need to now the derivative at the onset points for the
        # boundary conditions

        yp0 = jax.grad(left_fn)(x0)
        ypn = jax.grad(right_fn)(xn)

        # The onset points are appended to the provided points

        x = jnp.concatenate((jnp.array([x0]), x, jnp.array([xn])))
        y = jnp.concatenate((jnp.array([y0]), y, jnp.array([yn])))

        # We also need to append the new inner slopes

        h0 = x[1] - x[0]
        hn = x[-1] - x[-2]

        k0 = y[1] - y[0]
        kn = y[-1] - y[-2]

        s0 = k0 / h0
        sn = kn / hn

        h = jnp.concatenate((jnp.array([h0]), h, jnp.array([hn])))
        k = jnp.concatenate((jnp.array([k0]), k, jnp.array([kn])))
        s = jnp.concatenate((jnp.array([s0]), s, jnp.array([sn])))

        extrapolation_type = "custom"
        extrapolation_fn = (left_fn, right_fn)
      else:
        print("The extrapolation {} is unknown".format(extrapolation))

      p = (s[0:-1] * h[1:] + s[1:] * h[0:-1]) / (h[0:-1] + h[1:])

      # Build coefficient pairs

      s0s1 = s[0:-1] * s[1:]
      a = jnp.sign(s[0:-1])

      cond1 = jnp.logical_or(jnp.abs(p) > 2 * jnp.abs(s[0:-1]),
                            jnp.abs(p) > 2 * jnp.abs(s[1:]))

      tmp = jnp.where(cond1, 2 * a * jnp.where(jnp.abs(s[0:1]) > jnp.abs(s[1:]),
                                             jnp.abs(s[1:]), jnp.abs(s[0:-1])), p)

      slopes = jnp.where(s0s1 <= 0, 0.0, tmp)

      slopes = jnp.concatenate((jnp.array([yp0]), slopes, jnp.array([ypn])))

      # Build the coefficients and store properties

      a = (slopes[0:-1] + slopes[1:] - 2 * s) / jnp.square(h)
      b = (3 * s - 2 * slopes[0:-1] - slopes[1:]) / h
      c = slopes
      d = y[0:-1]

      coefficients = (a, b, c, d, extrapolation_args)

    self.x = x
    self.y = y
    self.coefficients = coefficients
    self.extrapolation_type = extrapolation_type
    self.extrapolation_fn = extrapolation_fn

  def __call__(self, x_new: Array) -> Array:
    """Evaluate spline at new data points.

    Args:
        x_new: Evaluation points

    Returns:
        Returns the interpolated values y_new corresponding to y_new.

    """

    a, b, c, d, extrapolation_args = self.coefficients

    # Find the interval of socket points, in which the new points lie

    x_new_idx = jnp.searchsorted(self.x, x_new, side="right") - 1

    # Find the out of bound values

    x_left_outside = x_new_idx < 0
    x_right_outside = x_new_idx > (self.x.size - 2)

    x_new_idx = jnp.where(x_left_outside, 0, x_new_idx)
    x_new_idx = jnp.where(x_right_outside, self.x.size - 2, x_new_idx)

    # Select the coefficients

    a = a[x_new_idx]
    b = b[x_new_idx]
    c = c[x_new_idx]
    d = d[x_new_idx]

    x = self.x[x_new_idx]

    # Interpolate the inner points

    y_new = a * jnp.power(x_new - x, 3) + b * jnp.power(x_new - x, 2) + c * (
        x_new - x) + d

    if self.extrapolation_type == "repulsion":

      exp, sigma, epsilon0, epsilon1, off = extrapolation_args

      # Set the extrapolation coefficient to zero if x must not be extrapolated.

      eps0 = jnp.where(x_left_outside, epsilon0, 0.0)
      eps1 = jnp.where(x_right_outside, epsilon1, 0.0)
      offset = jnp.where(x_left_outside, off, 0.0)

      # Find the points for which x must be interpolated, set the rest to 0
      # (extrapolation)

      inside = jnp.logical_and(jnp.logical_not(x_left_outside),
                              jnp.logical_not(x_right_outside))

      # Limit value of x to minimum to disallow division by zero

      x_new_nonzero = jnp.where(jnp.abs(x_new) < 1e-7, jnp.sign(x_new) * 1e-7,
                               x_new)

      extrapol = (eps0 + eps1) * jnp.power(sigma / x_new_nonzero, exp) + offset

      # Set extrapolated values

      y_new = extrapol + jnp.where(inside, y_new, 0.0)

    elif self.extrapolation_type == "custom":

      left_fn, right_fn = self.extrapolation_fn

      inside = jnp.logical_and(jnp.logical_not(x_left_outside),
                              jnp.logical_not(x_right_outside))

      left_extrapol = jnp.where(x_left_outside, left_fn(x_new), 0.0)
      right_extrapol = jnp.where(x_right_outside, right_fn(x_new), 0.0)

      # We need to cast the values to the right dtype given by the x_values

      left_extrapol = jnp.array(left_extrapol, dtype=x.dtype)
      right_extrapol = jnp.array(right_extrapol, dtype=x.dtype)

      y_new = left_extrapol + right_extrapol + jnp.where(inside, y_new, 0.0)

    return y_new

def tabulated(dr: Array, spline: Callable[[Array], Array],
              **unused_kwargs) -> Array:
  """
  Tabulated radial potential between particles given a spline function.

  Args:
      dr: An ndarray of pairwise distances between particles
      spline: A function computing the spline values at a given pairwise distance

  Returns:
      Array of energies
  """

  return spline(dr)

def tabulated_neighbor_list(displacement_or_metric: DisplacementOrMetricFn,
                            x_vals: Array,
                            y_vals: Array,
                            box_size: Box,
                            degree: int = 3,
                            r_onset: Array = 0.9,
                            r_cutoff: Array = 1.,
                            dr_threshold: Array = 0.2,
                            species: Array = None,
                            capacity_multiplier: float = 1.25,
                            initialize_neighbor_list: bool = True,
                            per_particle: bool = False) -> Callable[
  [Array], Array]:
  """
  Convenience wrapper to compute tabulated energy using a neighbor list.

  Provides option not to initialize neighborlist. This is useful if energy function needs
  to be initialized within a jitted function.
  """

  x_vals = jnp.array(x_vals, f32)
  y_vals = jnp.array(y_vals, f32)
  box_size = jnp.array(box_size, f32)
  r_onset = jnp.array(r_onset, f32)
  r_cutoff = jnp.array(r_cutoff, f32)
  dr_threshold = jnp.array(dr_threshold, f32)

  # Note: cannot provide the spline parameters via kwargs because only per-perticle parameters are supported
  spline = custom_interpolate.MonotonicInterpolate(x_vals, y_vals)
  tabulated_partial = partial(tabulated, spline=spline)

  energy_fn = smap.pair_neighbor_list(
    multiplicative_isotropic_cutoff(tabulated_partial, r_onset, r_cutoff),
    space.canonicalize_displacement_or_metric(displacement_or_metric),
    species=species,
    reduce_axis=(1,) if per_particle else None)

  if initialize_neighbor_list:
    neighbor_fn = partition.neighbor_list(displacement_or_metric, box_size,
                                          r_cutoff, dr_threshold,
                                          capacity_multiplier=capacity_multiplier)
    return neighbor_fn, energy_fn
  return energy_fn
