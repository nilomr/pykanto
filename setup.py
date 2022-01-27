#!/usr/bin/env/python
"""
Installation script
"""


import os
from setuptools import find_packages, setup

# TODO
LONG_DESCRIPTION = """
Package description
"""

if os.environ.get("READTHEDOCS", False) == "True":
    INSTALL_REQUIRES = []
    EXTRAS_REQUIRES = []
else:
    INSTALL_REQUIRES = [
        "pandas >= 1",
        "numpy >= 1.17",
        "scipy >= 1.5",
        "numba >= 0.49",
        "tqdm",
        "audio-metadata >= 0.9",
        "ray[default]",
        "pysoundfile >= 0.9",
        "umap-learn >= 0.5",
        "hdbscan >= 0.8",
        "seaborn >= 0.11",
        "scikit-image >= 0.18",
        "librosa >= 0.8",
        "bokeh >= 2.3.3",
        "ujson",
        "psutil"
    ]

    EXTRAS_REQUIRES = {
        'dev': [
            'sphinx',
            'sphinx-copybutton',
            'sphinx-rtd-theme'
        ]
    }

setup(
    name="pykanto",
    version="0.1.3",
    description="Management and analysis of animal vocalisation data",
    license="MIT",
    author='Nilo M. Recalde',
    author_email="nilomerinorecalde@gmail.com",
    url="https://github.com/nilomr/pykanto",
    long_description=LONG_DESCRIPTION,
    packages=["pykanto",
              "pykanto.signal",
              "pykanto.utils",
              "pykanto.intlabel",
              "pykanto.utils"
              ],
    package_data={
        'pykanto':
        ["data/segmented/*/*/*.wav", "data/segmented/*/*/*.JSON"]
    },
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRES,
    include_package_data=True,
)
