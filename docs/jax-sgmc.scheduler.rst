jax-sgmc.scheduler
=======================


.. automodule:: jax_sgmc.scheduler


Scheduler
----------

.. autofunction:: jax_sgmc.scheduler.init_scheduler


Specific Schedulers
--------------------

Step-size
__________

.. autosummary::
    :toctree: _autosummary

    polynomial_step_size
    cyclic_step_size

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
    init_burn_in