import io
import os
import setuptools

INSTALL_REQUIRES = [
    'numpy',
    'jax>=0.1.73',
    'jaxlib>=0.1.52',
    'dataclasses'
]

EXTRAS_REQUIRE = {
    'tensorflow' : [
        'tensorflow',
        'tensorflow_datasets'
    ],
    'test': [
        'pylint',
        'pytest',
        'pytest-mock'
    ],
    'docs': [
        'sphinx >= 3',
        'sphinx_rtd_theme',
        'sphinx-autodoc-typehints==1.11.1',
        'myst-nb',
        'sphinxcontrib-mermaid'
    ],
    'hdf5': [
        'h5py'
    ]
}

setuptools.setup(
    name='jax_sgmc',
    version='0.0.2',
    license='Apache 2.0',
    author='MMFM',
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
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
