from setuptools import setup, find_packages


description = "Package for the analysis of stochastic processes on random graphs with information theory."

setup(
    name="midynet",
    version=0.1,
    author="Charles Murphy",
    author_email="charles.murphy.1@ulaval.ca",
    url="https://github.com/charlesmurphy1/midynet",
    license="MIT",
    description=description,
    packages=find_packages(),
    install_requires=[
        "pybind11>=2.3",
        "numpy>=1.20.3",
        "scipy>=1.7.1",
        "psutil>=5.8.0",
        "tqdm>=4.56.0",
        "networkx>=2.8",
        "netrd",
        "graphinf>=0.1.0",
    ],
    python_requires=">=3.6",
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "midynet-cmd=midynet.cli:main",
        ],
    },
)
