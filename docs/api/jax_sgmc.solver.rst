jax_sgmc.solver
================

.. automodule:: jax_sgmc.solver


MCMC
-----

Run multiple chains of a solver in parallel or vectorized and save the results.

.. autoclass:: jax_sgmc.solver.mcmc


Solvers
-------

.. autofunction:: jax_sgmc.solver.sgmc
.. autofunction:: jax_sgmc.solver.amagold
.. autofunction:: jax_sgmc.solver.sggmc
.. autofunction:: jax_sgmc.solver.parallel_tempering

Solver States
--------------

.. autoclass:: jax_sgmc.solver.AMAGOLDState
.. autoclass:: jax_sgmc.solver.SGGMCState
