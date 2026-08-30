# rlt-python: compile the C++ core and pybind11 bindings with setup.py
# System deps: armadillo, lapack, blas, openmp, pybind11 (pip).
import os
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

SRC = os.path.join("src", "cpp")
BIND = os.path.join("src", "bindings")
INCLUDE = os.path.join(SRC, "include")

core_sources = []
for root, dirs, files in os.walk(SRC):
    for f in sorted(files):
        if f.endswith(".cpp"):
            core_sources.append(os.path.join(root, f))
for root, dirs, files in os.walk(BIND):
    for f in sorted(files):
        if f.endswith(".cpp"):
            core_sources.append(os.path.join(root, f))

ext = Pybind11Extension(
    "rlt._core",
    sources=core_sources,
    include_dirs=[SRC, BIND, INCLUDE],
    language="c++",
    cxx_std=17,
    extra_compile_args=["-O2", "-fopenmp", "-DARMA_DONT_PRINT_ERRORS",
                        "-DARMA_WARN_LEVEL=1", "-DARMA_DONT_USE_WRAPPER"],
    extra_link_args=["-fopenmp", "-llapack", "-lblas"],
)

setup(ext_modules=[ext], cmdclass={"build_ext": build_ext})
