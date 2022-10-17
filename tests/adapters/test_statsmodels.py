from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mlstars.adapters.statsmodels import ARIMA


@patch('statsmodels.tsa.arima.model.ARIMA')
def test_arima_1d(arima_mock):
    arima = ARIMA(1, 0, 0, 'ct', 3)
    X = np.array([1, 2, 3, 4, 5])
    arima.predict(X)
    assert_allclose(arima_mock.call_args[0][0], [1, 2, 3, 4, 5])
    assert arima_mock.call_args[1] == {'order': (1, 0, 0), 'trend': 'ct'}


def test_predict_1d():
    arima = ARIMA(1, 0, 0, 'ct', 3)

    X = np.array([1, 2, 3, 4, 5])
    result = arima.predict(X)

    expected = np.array([6, 7, 8])
    assert_allclose(result, expected, rtol=1e-4)


@patch('statsmodels.tsa.arima.model.ARIMA')
def test_arima_2d(arima_mock):
    arima = ARIMA(1, 0, 0, 'ct', 3)
    X = np.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
    ])
    arima.predict(X)
    assert_allclose(arima_mock.call_args_list[0][0], [[1, 2, 3, 4, 5]])
    assert_allclose(arima_mock.call_args_list[1][0], [[6, 7, 8, 9, 10]])
    assert_allclose(arima_mock.call_args_list[2][0], [[11, 12, 13, 14, 15]])
    assert arima_mock.call_args_list[0][1] == {'order': (1, 0, 0), 'trend': 'ct'}
    assert arima_mock.call_args_list[1][1] == {'order': (1, 0, 0), 'trend': 'ct'}
    assert arima_mock.call_args_list[2][1] == {'order': (1, 0, 0), 'trend': 'ct'}


def test_predict_2d():
    arima = ARIMA(1, 0, 0, 'ct', 3)

    X = np.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
    ])
    result = arima.predict(X)

    expected = np.array([
        [6, 7, 8],
        [11, 12, 13],
        [16, 17, 18]
    ])
    assert_allclose(result, expected, rtol=1e-4)


@patch('statsmodels.tsa.arima.model.ARIMA')
def test_arima_3d(arima_mock):
    arima = ARIMA(1, 0, 0, 'ct', 3)
    X = np.ones(shape=(3, 2, 1))
    with pytest.raises(ValueError):
        arima.predict(X)
