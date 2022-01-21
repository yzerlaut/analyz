from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='analyz',
    version='1.0',
    description='Common data analysis tools',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/yzerlaut/analyz',
    author='Yann Zerlaut',
    author_email='yann.zerlaut@cnrs.fr',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
    keywords='data analysis scipy numpy',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "h5py",
        "scikit-learn",
        "argparse"
    ]
)
