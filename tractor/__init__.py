from __future__ import absolute_import, unicode_literals
import os
import re
import jieba
import pickle
import random
from .feature import *
from .model import *
from .predict import *

from collections import defaultdict

__version__ = '0.10'
__license__ = 'MIT'

"""
## Text Classification

>>> from tractor import Tractor
# Tractor module
>>> Tr = Tractor('Sample')
>>> train_source = [('This is a Apple', 'fruit'),
                   ('This is a Banana', 'fruit'),
                   ('This is a Tomato', 'vegetable'),
                   ('This is a Capsicum', 'vegetable'),
                   ]
>>> Tr.train(train_source)
>>> test_source = [('This is a Apple', 'fruit'),
                   ('This is a Banana', 'fruit'),
                   ('This is a Tomato', 'vegetable'),
                   ('This is a Capsicum', 'vegetable'),
                   ]
>>> Tr.predict(test_source)
>>> Tr.save()
>>> Tr.load()

"""


# TRACTOR
class Tractor(object):
    def __init__(self, name, document_tokenize):
        self.name = name
        self.model = None
        self.train = False
        self.train_status = False
        self.load_status = False
        self.feature = None
        self.train_size = 0
        self.predict_size = 0
        self.train_source = None
        self.predict_result = None
        self.random_id = str(random.random())
        if document_tokenize is not None and not hasattr(document_tokenize, '__call__'):
            raise TractorException('Tokenize function must be callable.')
        self.document_tokenize = document_tokenize

    def get_load_status(self):
        return self.model is not None

    def train(self, train_source, delimiter='\t'):
        pass

    def predict(self, predict_source):
        pass

    def test(self, test_source, delimiter='\t'):
        pass

    def save(self, save_file):
        pass

    def load(self, load_file):
        pass


class TractorException(Exception):
    pass