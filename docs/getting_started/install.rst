.. _install:

.. highlight:: shell

Installation
============

Requirements
------------

Python
~~~~~~
**ml-stars** has been tested on **GNU/Linux**, and **macOS** systems running `Python 3.8, 3.9, 3.10, or 3.11`_ installed.

Also, although it is not strictly required, the usage of a `virtualenv`_ is highly recommended in
order to avoid having conflicts with other software installed in the system where you are trying to run **ml-stars**.

Install using pip
-----------------

The easiest and recommended way to install **ml-stars** is using `pip`_:

.. code-block:: console

    pip install ml-stars

This will pull and install the latest stable release from `PyPI`_.

Install from source
-------------------

The source code of **ml-stars** can be downloaded from the `Github repository`_

You can clone the repository and install it from source by running ``make install`` on the
``stable`` branch:

.. code-block:: console

    git clone git://github.com/sintel-dev/ml-stars
    cd ml-stars
    make install

If you are installing **ml-stars** in order to modify its code, the installation must be done
from its sources, in the editable mode, and also including some additional dependencies in
order to be able to run the tests and build the documentation. Instructions about this process
can be found in the :ref:`contributing` guide.

.. _Python 3.8, 3.9, 3.10, or 3.11: https://docs.python-guide.org/starting/installation/
.. _virtualenv: https://virtualenv.pypa.io/en/latest/
.. _pip: https://pip.pypa.io
.. _PyPI: https://pypi.org/
.. _Github repository: https://github.com/sintel-dev/Orion