import numpy as np
import h5py

filename = '/home/student/ana/jax-sgmc/examples/cifar100/results'

hf = h5py.File(filename, 'r')
print(hf['chain~0'].keys())
print(hf['chain~0']['likelihood'].shape)
print(hf['chain~0']['variables']['w'].shape)
i=0
