from __future__ import absolute_import, unicode_literals

__version__ = '0.10'
__license__ = 'MIT'

import random

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


class Tractor(object):
    def __init__(self, name):
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

    def train(self, document, delimiter='\t'):
        self.train_source = TractorTextSource(method='TRAIN', document=document)
        self.feature = TractorProcess(self.train_source).feature


class TractorModel(object):
    def __init__(self):
        pass


class TractorSVMModel(TractorModel):
    pass


class TractorProcess(object):
    def __init__(self, source):
        self.source = source

    @staticmethod
    def feature():
        return TractorFeature()


class TractorTextSource(object):
    def __init__(self, method, document):
        if method == 'TRAIN':
            for i, j in enumerate(document):
                self.id = i
                self.text = j[0]
                self.label = j[1]
                self.property = 'TRAIN'
        elif method == 'PREDICT':
            for i, j in enumerate(document):
                self.id = i
                self.text = j[0]
                self.label = None
                self.property = 'PREDICT'
        else:
            raise Exception('METHOD ERROR')


class TractorFeature(object):
    def __init__(self):
        pass


class TractorPredictResult(object):
    def __init__(self):
        pass
