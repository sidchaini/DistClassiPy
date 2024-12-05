.. DistClassiPy documentation master file, created by
   sphinx-quickstart on Wed Feb  7 16:42:22 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DistClassiPy's documentation!
========================================

DistClassiPy is a Python package for a distance-based classifier which can use several different distance metrics.

Installation
------------

To install DistClassiPy, run the following command:

.. code-block:: bash

    pip install distclassipy

Quick Start
-----

Here's a quick start guide to using DistClassiPy:

.. code-block:: python

    import distclassipy as dcpy
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1000,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=0,
        shuffle=False,
    )
    clf = dcpy.DistanceMetricClassifier(metric="canberra")
    clf.fit(X, y)
    print(clf.predict([[0, 0, 0, 0]]))

Features
--------

- Multiple distance metrics support
- Easy integration with existing data processing pipelines
- Efficient and scalable for large datasets

Authors
-------

Siddharth Chaini, Ashish Mahabal, Ajit Kembhavi and Federica B. Bianco.



Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

API Documentation
-----------------

.. toctree::
   :maxdepth: 1
   :caption: Get Started
   
   tutorial

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   classifier
   distances

.. toctree::
   :maxdepth: 1
   :caption: Index

   genindex
