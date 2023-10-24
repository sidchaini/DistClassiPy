from setuptools import setup, find_packages

setup(
    name="distclassipy",
    version="0.0.1",
    packages=find_packages(),
    author="Siddharth Chaini",
    author_email="sidchaini@gmail.com",
    description="A python package for a distance-based classifier which can use several different distance metrics.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sidchaini/DistClassiPy",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
    ],
)