#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name='aml-lab',
    packages=find_packages(),
    install_requires=[
        "tensorflow",
        "matplotlib",
        "pandas"
    ]
)