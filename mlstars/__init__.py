# -*- coding: utf-8 -*-

__author__ = 'MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com'
__version__ = '0.2.4.dev0'

import os

from mlblocks import MLBlock

MLBLOCKS_PRIMITIVES = os.path.join(os.path.dirname(__file__), 'primitives')
MLBLOCKS_PIPELINES = os.path.join(os.path.dirname(__file__), 'pipelines')


def load_primitive(primitive, arguments=None):
    arguments = arguments or dict()

    return MLBlock(primitive, **arguments)
