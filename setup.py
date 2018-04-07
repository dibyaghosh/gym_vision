# setup.py
from setuptools import setup,find_packages

setup(
    name='gym_vision',
    packages=[package for package in find_packages()
                if package.startswith('gym_vision')],
    version='0.1.0',
)
