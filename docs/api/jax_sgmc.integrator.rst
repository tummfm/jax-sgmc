jax_sgmc.integrator
====================

Overview
---------

.. automodule:: jax_sgmc.integrator

Integrators
------------

.. autofunction:: jax_sgmc.integrator.obabo
.. autofunction:: jax_sgmc.integrator.reversible_leapfrog
.. autofunction:: jax_sgmc.integrator.friction_leapfrog
.. autofunction:: jax_sgmc.integrator.langevin_diffusion

Integrator States
------------------

.. autoclass:: jax_sgmc.integrator.ObaboState
.. autoclass:: jax_sgmc.integrator.LeapfrogState
.. autoclass:: jax_sgmc.integrator.LangevinState

Utility
-------

.. autofunction:: jax_sgmc.integrator.random_tree
.. autofunction:: jax_sgmc.integrator.init_mass
