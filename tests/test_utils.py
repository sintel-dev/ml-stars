# -*- coding: utf-8 -*-

from mlstars.utils import import_object


class Dummy(object):
	def some_function(self):
		pass


def test_import_object():
    imported_dummy = import_object(__name__ + '.Dummy')

    assert Dummy is imported_dummy


def test_import_object_grand_parent():
	imported_function = import_object(__name__ + '.Dummy.some_function')

	assert Dummy.some_function is imported_function
