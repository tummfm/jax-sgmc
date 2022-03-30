jax_sgmc.scheduler
=======================

.. automodule:: jax_sgmc.scheduler

Base Scheduler
--------------

.. autofunction:: jax_sgmc.scheduler.init_scheduler

.. autofunction:: jax_sgmc.scheduler.scheduler_state

.. autofunction:: jax_sgmc.scheduler.schedule

.. autofunction:: jax_sgmc.scheduler.static_information

Specific Schedulers
-------------------

.. autoclass:: jax_sgmc.scheduler.specific_scheduler

Step-size
__________

.. autosummary::
    :toctree: _autosummary

    polynomial_step_size
    polynomial_step_size_first_last
    adaptive_step_size

Temperature
____________

.. autosummary::
    :toctree: _autosummary

    constant_temperature
    cyclic_temperature

Burn In
________

.. autosummary::
    :toctree: _autosummary

    cyclic_burn_in
    initial_burn_in

Thinning
__________

.. autosummary::
    :toctree: _autosummary

    random_thinning

