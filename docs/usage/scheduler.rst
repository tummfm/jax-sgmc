Setup Schedulers
==================

A scheduler is a combination of specific schedulers, which control only a single
parameter, for example the step size.
Specific schedulers for different variables are then combined into a basic
scheduler via :func:`jax_sgmc.scheduler.init_scheduler`, which updates all
specific schedulers and provides default values for parameters without a
specific scheduler.


Specific Schedulers
--------------------

.. doctest::

  >>> from jax_sgmc import scheduler
  >>>
  >>> step_size_schedule_unused = scheduler.polynomial_step_size(
  ...   a=0.1, b=1.0, gamma=0.33)

We already provided all required arguments. However, it is also possible to
provide only the arguments, which should stay equal over all chains.
For example we could provide different ``gamma``-values by specifying them
during the initialization of the basic scheduler:

  >>> step_size_schedule_partial = scheduler.polynomial_step_size(
  ...   a=0.1, b=1.0)

Basic Scheduler
---------------

It is not necessary to setup a scheduler for all parameters, because the basic
scheduler provides default values.
Therefore, we can initialize the basic scheduler only with the specific step
size schedule we initialized above:

  >>> init_fn, next_fn, get_fn = scheduler.init_scheduler(
  ...   step_size=step_size_schedule_partial, progress_bar=False)


After we created the basic scheduler, we can initialize a schedule.
Here we have to provide the missing values for the partially initialized
schedulers.

  >>> sched_a, static_information = init_fn(10, step_size={'gamma': 0.1})
  >>> sched_b, _ = init_fn(10, step_size={'gamma': 1.0})

Static information is returned in addition to the scheduler state, e.g.
the total number of iterations or the expected number of collected samples.
This information is necessary, e.g., for the ``io``-module to allocate
sufficient memory for the samples to be saved.

  >>> print(static_information)
  static_information(samples_collected=10)

In this example, we can see that the temperature parameter has been assigned to
a default value of 1.0 and the different step size schedules are updated with
different gamma parameters:

  >>> curr_sched_a = get_fn(sched_a)
  >>> curr_sched_b = get_fn(sched_b)

  >>> print(f"Scheduler a\n===========\n"
  ...       f"  Step-Size = {curr_sched_a.step_size : .2f}\n"
  ...       f"  Temperature = {curr_sched_a.temperature : .2f}")
  Scheduler a
  ===========
    Step-Size =  0.10
    Temperature =  1.00
  >>> print(f"Scheduler b\n===========\n"
  ...       f"  Step-Size = {curr_sched_b.step_size : .2f}\n"
  ...       f"  Temperature = {curr_sched_b.temperature : .2f}")
  Scheduler b
  ===========
    Step-Size =  0.10
    Temperature =  1.00

  >>> # Get the parameters at the next iteration
  >>> sched_a = next_fn(sched_a)
  >>> sched_b = next_fn(sched_b)
  >>> curr_sched_a = get_fn(sched_a)
  >>> curr_sched_b = get_fn(sched_b)

  >>> print(f"Scheduler a\n===========\n"
  ...       f"  Step-Size = {curr_sched_a.step_size : .2f}\n"
  ...       f"  Temperature = {curr_sched_a.temperature : .2f}")
  Scheduler a
  ===========
    Step-Size =  0.09
    Temperature =  1.00
  >>> print(f"Scheduler b\n===========\n"
  ...       f"  Step-Size = {curr_sched_b.step_size : .2f}\n"
  ...       f"  Temperature = {curr_sched_b.temperature : .2f}")
  Scheduler b
  ===========
    Step-Size =  0.05
    Temperature =  1.00
