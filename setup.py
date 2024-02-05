from setuptools import setup, find_packages

setup(
    name="distclassipy",
    version="0.0.2",
    packages=find_packages(),
    author="Siddharth Chaini",
    author_email="sidchaini@gmail.com",
    description="A python package for a distance-based classifier which can use several different distance metrics.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sidchaini/DistClassiPy",
    # classifier keywords: https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13"
    ],
    install_requires=[
        "joblib>=1.3.2",
        "numpy>=1.26.3",
        "pandas>=2.2.0",
        "scikit-learn>=1.4.0",
        "scipy>=1.12.0"
    ],
)