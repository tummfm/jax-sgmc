jax_sgmc.scheduler
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
    polynomial_step_size_first_last

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