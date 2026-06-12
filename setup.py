import codecs
import os.path

from setuptools import setup, Extension

from Cython.Build import cythonize


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


extensions = [
    Extension("distclassipy._cdistances", ["distclassipy/_cdistances.pyx"]),
]

setup(
    version=get_version("distclassipy/__init__.py"),
    ext_modules=cythonize(extensions, language_level="3"),
)
