.. _quickstart:

Quickstart
==========

In the following steps we will show a short guide to help you getting started with ml-stars.

We will be executing a single primitive for data transformation.

1. Load the primitive
---------------------

The first step in order to run a primitive is to load it.

This will be done using the ``mlstars.load_primitive`` function, which will
load the indicated primitive as an `MLBlock Object from MLBlocks`_.

In this case, we will load the ``sklearn.preprocessing.MinMaxScaler`` primitive.

.. ipython:: python
    :okwarning:

	from mlstars import load_primitive

	primitive = load_primitive('sklearn.preprocessing.MinMaxScaler')


2. Load some data
-----------------

The StandardScaler is a transformation primitive which scales your data into a given range.

To use this primtives, we generate a synthetic data with some numeric values.

.. ipython:: python
    :okwarning:

	import numpy as np

	data = np.array([10, 1, 3, -1, 5, 6, 0, 4, 13, 4]).reshape(-1, 1)


The ``data`` is a list of integers where their original range is between [-1, 13].


3. Fit the primitive
--------------------

In order to run our primitive, we first need to fit it.

This is the process where it analyzes the data to detect what is the original range of the data.

This is done by calling its `fit` method and passing the `data` as `X`.

.. ipython:: python
    :okwarning:

	primitive.fit(X=data)


4. Produce results
------------------

Once the pipeline is fit, we can process the data by calling the `produce` method of the
primitive instance and passing agin the `data` as `X`.

.. ipython:: python
    :okwarning:

	transformed = primitive.produce(X=data)
	transformed


We can see how the ``transformed`` data contains the transformed values and the data 
is now in [0, 1] range.

.. _MLBlock Object from MLBlocks: https://MLBazaar.github.io/MLBlocks/api/mlblocks.html#mlblocks.MLBlock
