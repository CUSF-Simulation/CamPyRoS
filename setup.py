"""
Allows installation via pip by navigating to this directory, and running "pip install ."
"""

import sys
from setuptools import setup, find_packages

setup(
    name="CamPyRoS",
    version="1.1",
    author="Jago Strong-Wright & Daniel Gibbons",
    author_email="jagoosw@protonmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.1",
        "matplotlib",
        "scipy",
        "ambiance",
        "thermo",
        "gas_dynamics",
        "pandas",
        "numexpr",
        "requests",
        "getgfs>=1.0.0"
    ],
    description="Cambridge Python Rocketry Simulator",
)
