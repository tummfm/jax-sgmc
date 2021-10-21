import io
import os
import setuptools

INSTALL_REQUIRES = [
    'numpy',
    'jax>=0.1.73,<0.2.21',
    'jaxlib>=0.1.52,<0.1.72',
    'dataclasses'
]

setuptools.setup(
    name='jax_sgmc',
    version='0.0.1',
    license='Apache 2.0',
    author='MMFM',
    install_requires=INSTALL_REQUIRES,
    packages=setuptools.find_packages(),
    long_description_content_type='text/markdown',
    description='Stochastic Gradient Monte Carlo using Jax',
    python_requires='>=3.8',
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: MacOS',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
    ])
