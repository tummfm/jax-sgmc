Setup Solver by Combining Modules
==================================

Schedulers
-----------

A scheduler is a combination of specific schedulers, which control only a single
parameter, for example the step size.

.. doctest::

  >>> from jax_sgmc import scheduler
  >>>
  >>> step_size_schedule = scheduler.polynomial_step_size(
  ...   a=0.1, b=1.0, gamma=0.33)

We already provided all required arguments. However, it is also possible to
provide only the arguments, which should stay equal over all chains.

  >>> step_size_schedule_partial = scheduler.polynomial_step_size(
  ...   a=0.1, b=1.0)

For all other schedules, the scheduler provide default values. So if we are not
interested in applying burn in or thinning to the chain, we do not have to
initialize a scheduler.

  >>> init_fn, next_fn, get_fn = scheduler.init_scheduler(
  ...   step_size=step_size_schedule_partial)

Now we can provide different values for the partialy initialized schedulers.
In addition to the scheduler states, we also get a dict which contains
information such as the total count of accepted samples.

  >>> sched_a, static_information = init_fn(10, step_size={'gamma': 0.1})
  >>> sched_b, _ = init_fn(10, step_size={'gamma': 1.0})
  >>>
  >>> print(static_information)
  static_information(samples_collected=10)
  >>> print(get_fn(sched_a))
  schedule(step_size=DeviceArray(0.1, dtype=float32), temperature=DeviceArray(1., dtype=float32), burn_in=DeviceArray(1., dtype=float32), accept=DeviceArray(True, dtype=bool))
  >>> print(get_fn(sched_b))
  schedule(step_size=DeviceArray(0.1, dtype=float32), temperature=DeviceArray(1., dtype=float32), burn_in=DeviceArray(1., dtype=float32), accept=DeviceArray(True, dtype=bool))
  >>>
  >>> # Get the parameters at the next iteration
  >>> sched_a = next_fn(sched_a)
  >>> sched_b = next_fn(sched_b)
  >>>
  >>> print(get_fn(sched_a))
  schedule(step_size=DeviceArray(0.0933033, dtype=float32), temperature=DeviceArray(1., dtype=float32), burn_in=DeviceArray(1., dtype=float32), accept=DeviceArray(True, dtype=bool))
  >>> print(get_fn(sched_b))
  schedule(step_size=DeviceArray(0.05, dtype=float32), temperature=DeviceArray(1., dtype=float32), burn_in=DeviceArray(1., dtype=float32), accept=DeviceArray(True, dtype=bool))
