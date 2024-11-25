# -*- coding: utf-8 -*-

from sklearn.preprocessing import MinMaxScaler


class Scaler(MinMaxScaler):
    """sklearn.preprocessing.MinMaxScaler adapter.

    Convert feature_range into a tuple.

    Args:
        feature_range (list):
            List of two elements. Desired range of transformed data.

        copy (bool):
            Set to False to perform inplace row normalization and avoid a copy.
    """

    def __init__(self, feature_range=(0, 1), copy=True):
        feature_range = tuple(feature_range)

        super(Scaler, self).__init__(
            feature_range=feature_range,
            copy=copy
        )
