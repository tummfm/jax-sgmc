Extending Schedulers
=====================

Integration with Base Scheduler
--------------------------------

Global and Local Scheduler Arguments
_____________________________________

- *Global only* arguments are provided by position to the scheduler
- *Global and local* arguments are provided by keyword to the scheduler and the
  init function such that they can be overwritten.

For example:

::

  def some_scheduler(global_arg, global_or_local=0.0):

    def init_fn(global_or_local = global_or_local):
      # Use local arg
    ...
