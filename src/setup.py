from distutils.core import setup

import setuptools

setup(
    name='trader',
    version='0.2.7',
    description='Almanaque Trader',
    author='J. Renero, J. Gonzalez',
    packages=setuptools.find_packages(),
    url='https://github.com/renero/trader',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
