================
Installing pypkg
================

Python version
--------------

The package :code:`pypkg` is written in Python
and requires Python 3.10 or later.

:code:`pypkg` package is distributed at
`PyPI <https://pypi.org/project/pypkg/>`_.
To install the package:

.. code::

   pip install pypkg

For developers, you can clone the repository and install the package in the
development mode.

.. code::

    git clone https://github.com/ihmeuw-msca/pypkg.git
    cd pypkg
    pip install -e ".[test,docs]"