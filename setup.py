import os.path

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

# The directory containing this file
HERE = os.path.abspath(os.path.dirname(__file__))

VERSION='1.0.3'
# VERSION='0.0.1'

# The text of the README file
with open(os.path.join(HERE, "README.md")) as fid:
    README = fid.read()

setup(
    name='polycracker',
    version=VERSION,
    author='polycracker team',
    author_email='sgordon@lbl.gov',
    description='unsupervised classification of polyploid subgenomes',
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://bitbucket.org/berkeleylab/jgi-polycracker",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
    ],
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "polycracker=polycracker.polycracker:polycracker",
            "repeats=polycracker.repeats:repeats",
            "utilities=polycracker.utilities:utilities",
            "format=polycracker.format:format",
        ]
    },
)
