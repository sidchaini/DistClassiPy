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

Usage
-----

Here's a quick example to get you started with DistClassiPy:

.. code-block:: python

    import distclassipy as dcpy
    clf = dcpy.DistanceMetricClassifier()
    # Add your data and labels
    clf.fit(data, labels)
    # Predict new instances
    predictions = clf.predict(new_data)

Features
--------

- Multiple distance metrics support
- Easy integration with existing data processing pipelines
- Efficient and scalable for large datasets

Contact
-------

For any queries, please reach out to Siddharth Chaini at sidchaini@gmail.com.



Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

API Documentation
-----------------

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   classifier
   distances