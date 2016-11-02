#!/usr/bin/env python

import os
import sys

from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext as _build_ext


# workaround for installing some_sums when numpy is not present
# taken from:
# stackoverflow.com/questions/19919905/
# how-to-bootstrap-numpy-installation-in-setup-py#21621689
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


def prepare_modules():
    from some_sums.src.template import make_c_files
    make_c_files()
    platform = sys.platform
    if platform == "darwin":
        extra_compile_args = ['-O2', '-msse3', '-mavx', '-fopenmp']
        extra_link_args = ['-lomp']
    elif platform == "win32":
        extra_compile_args = ['-O2', '/openmp', '/arch:AVX']
        extra_link_args = []
    else:
        extra_compile_args = ['-O2', '-msse3', '-mavx', '-fopenmp']
        extra_link_args = ['-lgomp']
    ext = [Extension("some_sums.sums",
                     sources=["some_sums/src/sums.c"],
                     include_dirs=[],
                     extra_compile_args=extra_compile_args,
                     extra_link_args=extra_link_args)]
    return ext


def get_long_description():
    with open('README.rst', 'r') as fid:
        long_description = fid.read()
    idx = max(0, long_description.find("some_sums is a collection"))
    long_description = long_description[idx:]
    return long_description


def get_version_str():
    ver_file = os.path.join('some_sums', 'version.py')
    with open(ver_file, 'r') as fid:
        version = fid.read()
    version = version.split("= ")
    version = version[1].strip()
    version = version.strip("\"")
    return version


CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU General Public License v3 "
    "or later (GPLv3+)",
    "Operating System :: OS Independent",
    "Programming Language :: C",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering"]


metadata = dict(name='some_sums',
                maintainer="Keith Goodman",
                maintainer_email="bottle-neck@googlegroups.com",
                description="What's the fastest way to sum a NumPy array?",
                long_description=get_long_description(),
                url="https://github.com/kwgoodman/some_sums",
                license="GNU GPLv3+",
                classifiers=CLASSIFIERS,
                platforms="OS Independent",
                version=get_version_str(),
                packages=find_packages(),
                package_data={'some_sums': ['LICENSE']},
                requires=['numpy'],
                install_requires=['numpy'],
                cmdclass={'build_ext': build_ext},
                setup_requires=['numpy'])


if not(len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or
       sys.argv[1] in ('--help-commands', 'egg_info', '--version', 'clean',
                       'build_sphinx'))):
    # build some_sums
    metadata['ext_modules'] = prepare_modules()

setup(**metadata)
