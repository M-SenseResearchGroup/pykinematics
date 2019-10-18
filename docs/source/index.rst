.. pykinematics documentation master file, created by
   sphinx-quickstart on Fri Oct 18 10:03:25 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pykinematics: 3D hip joint angle estimation
===========================================
``pykinematics`` is an open-source Python package for estimating hip kinematics using both novel Magnetic and Inertial
Measurement Unit (MIMU) wearable sensors and existing Optical Motion Capture (OMC) algorithms. The novel MIMU algorithms
have been validated against OMC, and include novel methods for estimating sensor-to-sensor relative orientation and
sensor-to-segment alignment.

Validation
----------
The novel MIMU algorithms in ``pykinematics`` were validated against OMC in a healthy human subject population.
Detailed results and description of those novel algorithms can be found in [1]_.

License
-------
``pykinematics`` is open source software distributed under a GNU GPL-3.0 license.

Papers
------

.. [1] L. Adamowicz, R. Gurchiek, J. Ferri, A. Ursiny, N. Fiorentino, and R. McGinnis. "Novel Algorithms for Estimating Relative Orientation and Hip Joint Angles from Wearable Sensors." *Sensors*. Under-review.

Contents
--------
.. toctree::
   :maxdepth: 2

   installation
   usage
   ref/index



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
