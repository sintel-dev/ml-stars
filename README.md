<p align="left">
  <a href="https://dai.lids.mit.edu">
    <img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt="DAI-Lab" />
  </a>
  <i>An Open Source Project from the <a href="https://dai.lids.mit.edu">Data to AI Lab, at MIT</a></i>
</p>

[![Development Status](https://img.shields.io/badge/Development%20Status-2%20--%20Pre--Alpha-yellow)](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
[![PyPi Shield](https://img.shields.io/pypi/v/ml-stars.svg)](https://pypi.python.org/pypi/ml-stars)
[![Tests](https://github.com/sintel-dev/ml-stars/workflows/Run%20Tests/badge.svg)](https://github.com/sintel-dev/ml-stars/actions?query=workflow%3A%22Run+Tests%22+branch%3Amaster)
[![Downloads](https://pepy.tech/badge/ml-stars)](https://pepy.tech/project/ml-stars)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MLBazaar/MLBlocks/master?filepath=examples/tutorials)

# ml-stars

Primitives for machine learning and time series.

* Github: https://github.com/sintel-dev/ml-stars
* License: [MIT](https://github.com/sintel-dev/ml-stars/blob/master/LICENSE)
* Development Status: [Pre-Alpha](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)

# Overview

This repository contains primitive annotations to be used by the MLBlocks library, as well as
the necessary Python code to make some of them fully compatible with the MLBlocks API requirements.

There is also a collection of custom primitives contributed directly to this library, which either
combine third party tools or implement new functionalities from scratch.

# Installation

## Requirements

**ml-stars** has been developed and tested on [Python 3.8, 3.9, 3.10, 3.11, and 3.12](https://www.python.org/downloads/)

Also, although it is not strictly required, the usage of a
[virtualenv](https://virtualenv.pypa.io/en/latest/) is highly recommended in order to avoid
interfering with other software installed in the system where **ml-stars** is run.

## Install with pip

The easiest and recommended way to install **ml-stars** is using [pip](https://pip.pypa.io/en/stable/):

```bash
pip install ml-stars
```

This will pull and install the latest stable release from [PyPi](https://pypi.org/).

If you want to install from source or contribute to the project please read the
[Contributing Guide](https://github.com/sintel-dev/ml-stars/blob/master/CONTRIBUTING.rst).

# Quickstart

This section is a short series of tutorials to help you getting started with ml-stars.

We will be executing a single primitive for data transformation.

### 1. Load a Primitive

The first step in order to run a primitive is to load it.

This will be done using the `mlstars.load_primitive` function, which will
load the indicated primitive as an [MLBlock Object from MLBlocks](https://MLBazaar.github.io/MLBlocks/api/mlblocks.html#mlblocks.MLBlock)

In this case, we will load the `sklearn.preprocessing.MinMaxScaler` primitive.

```python3
from mlstars import load_primitive

primitive = load_primitive('sklearn.preprocessing.MinMaxScaler')
```

### 2. Load some data

The StandardScaler is a transformation primitive which scales your data into a given range.

To use this primtives, we generate a synthetic data with some numeric values.
```python3
import numpy as np

data = np.array([10, 1, 3, -1, 5, 6, 0, 4, 13, 4]).reshape(-1, 1)
```

The `data` is a list of integers where their original range is between [-1, 13].


### 3. Fit the primitive

In order to run our primitive, we first need to fit it.

This is the process where it analyzes the data to detect what is the original range of the data.

This is done by calling its `fit` method and passing the `data` as `X`.

```python3
primitive.fit(X=data)
```

### 4. Produce results

Once the pipeline is fit, we can process the data by calling the `produce` method of the
primitive instance and passing agin the `data` as `X`.

```python3
transformed = primitive.produce(X=data)
transformed
```

After this is done, we can see how the transformed data contains the transformed values:

```
array([[0.78571429],
       [0.14285714],
       [0.28571429],
       [0.        ],
       [0.42857143],
       [0.5       ],
       [0.07142857],
       [0.35714286],
       [1.        ],
       [0.35714286]])
```

The data is now in [0, 1] range.

## What's Next?

Documentation
