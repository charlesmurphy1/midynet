import os
import sys
import setuptools
import importlib
import pathlib
from setuptools import setup, find_packages


description = "Package for the analysis of stochastic processes on random graphs with information theory."
if importlib.util.find_spec("graphinf") is None:
    current_dir = os.getcwd()
    p = pathlib.Path("modules/graphinf/")
    os.chdir(p)
    print("Installing graphinf")
    os.system("pip install .")
    os.chdir(current_dir)


setup(
    name="midynet",
    version=0.1,
    author="Charles Murphy",
    author_email="charles.murphy.1@ulaval.ca",
    url="https://github.com/charlesmurphy1/fast-midynet",
    license="MIT",
    description=description,
    packages=find_packages(),
    install_requires=[
        "pybind11>=2.3",
        "numpy>=1.20.3",
        "scipy>=1.7.1",
        "psutil>=5.8.0",
        "tqdm>=4.56.0",
    ],
    python_requires=">=3.6",
    zip_safe=False,
)
