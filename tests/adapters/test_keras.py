# -*- coding: utf-8 -*-

import pickle
from unittest import TestCase

import numpy as np
import tensorflow as tf

from mlstars.adapters.keras import Sequential, build_layer

TEST_LAYER = {
    "class": "tensorflow.keras.layers.Dense",
    "parameters": {
        "units": 5
    }
}


TEST_LAYER_WRAPPER = {
    "class": "tensorflow.keras.layers.TimeDistributed",
    "parameters": {
        "layer": {
            "class": "tensorflow.keras.layers.Dense",
            "parameters": {
                "units": 1
            }
        }
    }
}


TEST_HYPERPARAMETER = {
    "class": "tensorflow.keras.layers.Dropout",
    "parameters": {
        "rate": "drop_rate"
    }
}


def test_build_layer():
    built_layer = build_layer(TEST_LAYER, None)

    assert isinstance(built_layer, tf.keras.layers.Dense)


def test_build_wrapper_layer():
    built_layer = build_layer(TEST_LAYER_WRAPPER, None)

    assert isinstance(built_layer, tf.keras.layers.TimeDistributed)


def test_build_layer_hyperparameters():
    built_layer = build_layer(TEST_HYPERPARAMETER, {"drop_rate": 0.1})

    assert isinstance(built_layer, tf.keras.layers.Dropout)
    assert built_layer.rate == 0.1


class SequentialTest(TestCase):
    @classmethod
    def setup_class(cls):
        cls.layers = [
            {
                "class": "tensorflow.keras.layers.Layer",
                "parameters": {}
            }
        ]

        cls.loss = "tensorflow.keras.losses.mean_squared_error"

        cls.optimizer = "tensorflow.keras.optimizers.Adam"

        cls.classification = False

        cls.input = np.random.rand(10, 5, 1)

        cls.output = cls.input

    def test__setdefault_in_kwargs(self):
        # Setup
        sequential = Sequential(None, None, None, None)

        # Run
        kwargs = {'input_shape': [100, 1]}
        sequential._setdefault(kwargs, 'input_shape', 'whatever')

        # Assert
        assert kwargs['input_shape'] == [100, 1]

    def test__setdefault_not_in_hyperparameters(self):
        # Setup
        sequential = Sequential(None, None, None, None)

        # Run
        kwargs = dict()
        sequential._setdefault(kwargs, 'input_shape', 'whatever')

        # Assert
        assert kwargs == dict()

    def test__setdefault_not_none(self):
        # Setup
        sequential = Sequential(None, None, None, None, input_shape=[100, 1])

        # Run
        kwargs = dict()
        sequential._setdefault(kwargs, 'input_shape', 'whatever')

        # Assert
        assert kwargs == dict()

    def test__setdefault_none(self):
        # Setup
        sequential = Sequential(None, None, None, None, input_shape=None)

        # Run
        kwargs = dict()
        sequential._setdefault(kwargs, 'input_shape', [100, 1])

        # Assert
        assert kwargs == {'input_shape': [100, 1]}

    def test__augment_hyperparameters_3d_numpy(self):
        # Setup
        sequential = Sequential(None, None, None, None, input_shape=None)

        # Run
        kwargs = dict()
        X = np.array([
            [[1, 2, 3, 4],
             [1, 2, 3, 4],
             [1, 2, 3, 4]],
            [[1, 2, 3, 4],
             [1, 2, 3, 4],
             [1, 2, 3, 4]],
        ])
        Sequential._augment_hyperparameters(sequential, X, 'input', kwargs)

        # Assert
        assert kwargs == {'input_shape': (3, 4)}

    def test_fit(self):
        # Setup
        sequential = Sequential(self.layers, self.loss, self.optimizer, self.classification)

        # Run
        sequential.fit(np.random.rand(10,), self.output)

    def test_fit_predict(self):
        # Setup
        sequential = Sequential(self.layers, self.loss, self.optimizer, self.classification)

        # Run
        sequential.fit(self.input, self.output)
        output = sequential.predict(self.input)

        # Assert
        np.testing.assert_array_almost_equal(output, self.output)

    def test_fit_predict_classification(self):
        # Setup
        sequential = Sequential(self.layers, self.loss, self.optimizer, classification=True)

        # Run
        sequential.fit(self.input, self.output)
        sequential.predict(self.input)

    def test_callback(self):
        # Setup
        callbacks = [{
            "class": "tensorflow.keras.callbacks.EarlyStopping"
        }]

        # Run
        Sequential(None, None, None, None, callbacks=callbacks)

    def test_save_load(self):
        # Setup
        sequential = Sequential(self.layers, self.loss, self.optimizer, self.classification)
        sequential.fit(self.input, self.output)
        path = 'some_path.pkl'

        # Run
        with open(path, 'wb') as pickle_file:
            pickle.dump(sequential, pickle_file)

        with open(path, 'rb') as pickle_file:
            new_sequential = pickle.load(pickle_file)

        # Assert
        assert isinstance(new_sequential, type(sequential))
