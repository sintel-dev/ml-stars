import warnings

import numpy as np
from statsmodels.tsa.arima import model

_ARIMA_MODEL_DEPRECATION_WARNING = (
    "statsmodels.tsa.arima_model.Arima is deprecated "
    "and will be removed in a future version. Please use "
    "statsmodels.tsa.arima.model.ARIMA instead."
)


class ARIMA(object):
    """A Wrapper for the statsmodels.tsa.arima.model.ARIMA class."""

    def __init__(self, p, d, q, trend, steps):
        """Initialize the ARIMA object.

        Args:
            p (int):
                Integer denoting the order of the autoregressive model.
            d (int):
                Integer denoting the degree of differencing.
            q (int):
                Integer denoting the order of the moving-average model.
            trend (str):
                Parameter controlling the deterministic trend. Can be specified
                as a string where 'c' indicates a constant term, 't' indicates
                a linear trend in time, and 'ct' includes both.
            steps (int):
                Integer denoting the number of time steps to predict ahead.
        """
        warnings.warn(_ARIMA_MODEL_DEPRECATION_WARNING, DeprecationWarning)

        self.p = p
        self.d = d
        self.q = q
        self.trend = trend
        self.steps = steps

    def predict(self, X):
        """Predict values using the initialized object.

        Args:
            X (ndarray):
                N-dimensional array containing the input sequences for the model.

        Returns:
            ndarray:
                N-dimensional array containing the predictions for each input sequence.
        """
        arima_results = list()
        dimensions = len(X.shape)

        if dimensions > 2:
            raise ValueError("Only 1D o 2D arrays are supported")

        if dimensions == 1 or X.shape[1] == 1:
            X = np.expand_dims(X, axis=0)

        num_sequences = len(X)
        for sequence in range(num_sequences):
            arima = model.ARIMA(X[sequence], order=(self.p, self.d, self.q), trend=self.trend)
            arima_fit = arima.fit()
            arima_results.append(arima_fit.forecast(self.steps)[0])

        arima_results = np.asarray(arima_results)

        if dimensions == 1:
            arima_results = arima_results[0]

        return arima_results
