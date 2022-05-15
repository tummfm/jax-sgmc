Saving of Samples
==================

Seting up Saving
-----------------

**jax_sgmc** supports saving and checkpointing inside of jit-compiled
functions. Saving data consists of two parts:

Data Collector
_______________

The data collector serializes the data and writes it to the disk and or keeps
it in memory. Every data collector following the interface works.

Saving
________

The function save initializes the interface between the data collector and the
jit-compiled function.

If the device memory is large, it is possible to use
:func:`jax_sgmc.io.no_save`. This function has the same signature towards the
jit-compiled function but keeps the all collected data in the device memory.


Extending Saveable PyTree Types
--------------------------------

By default, transformations are defined for some default types:

  - list
  - dict
  - (named)tuple

Additionaly, transformation for the following optional libraries are
implemented:

  - haiku._src.data_structures.FlatMapping

A new transformation rule is a function, which accepts a pytree node of
a specific type an returns a iterable, which itself returns `(key, value)`-
pairs.

.. doctest::

    >>> from jax_sgmc import io
    >>> from jax.tree_util import register_pytree_node
    >>>
    >>> class SomeClass:
    ...   def __init__(self, value):
    ...     self._value = value
    >>>
    >>> # Do not forget to register the class as jax pytree node
    >>> register_pytree_node(SomeClass,
    ...                      lambda sc: (sc._value, None),
    ...                      lambda _, data: SomeClass(value=data))
    >>>
    >>> # Now define a rule to transform the class into a dict
    >>> @io.register_dictionize_rule(SomeClass)
    ... def some_class_to_dict(instance_of_some_class):
    ...   return [("this_is_the_key", instance_of_some_class._value)]
    >>>
    >>> some_class = SomeClass({'a': 0.0, 'b': 0.5})
    >>> some_class_as_dict = io.pytree_to_dict(some_class)
    >>>
    >>> print(some_class_as_dict)
    {'this_is_the_key': {'a': 0.0, 'b': 0.5}}

