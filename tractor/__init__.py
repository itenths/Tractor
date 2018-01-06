from __future__ import absolute_import, unicode_literals
import os
import re
import jieba
import pickle
import random

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
        pass

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


# MODEL
class TractorModel(object):
    def __init__(self):
        pass


# SVM MODEL
class TractorSVMModel(TractorModel):
    pass


# PROCESS
class TractorProcess(object):
    def __init__(self, source):
        self.source = source

    @staticmethod
    def feature():
        return TractorFeature()


# TEXT SOURCE
def _dict2list(d):
    if len(d) == 0:
        return []

    m = max(v for k, v in d.items())
    index_list = [] * (m + 1)
    for k, v in d.items():
        index_list[v] = k
    return index_list


def _list2dict(l):
    return dict((v, k) for k, v in enumerate(l))


class TractorTextPreProcessor(object):
    def __init__(self):
        self.token2id = {'>>dummy<<': 0}
        self.id2token = None

    @staticmethod
    def _default_tokenize(text):
        return jieba.cut(text.strip(), cut_all=True)

    def preprocess(self, text, document_tokenize):
        if document_tokenize is not None:
            tokens = document_tokenize(text)
        else:
            tokens = self._default_tokenize(text)
        tokens_index = []
        for _, t in enumerate(tokens):
            if t not in self.token2id:
                self.token2id[t] = len(self.token2id)
            tokens_index.append(self.token2id[t])
        return tokens_index

    def save(self, save_file):
        self.id2token = _dict2list(self.token2id)
        config = {'id2token': self.id2token}
        pickle.dump(config, open(save_file, 'wb'), -1)

    def load(self, load_file):
        config = pickle.load(open(load_file, 'rb'))
        self.id2token = config['id2token']
        self.token2id = _list2dict(self.id2token)
        return self


class TractorFeatureGenerator(object):
    def __init__(self):
        self.ngram2id = {'>>dummy<<': 0}
        self.id2ngram = None

    def unigram(self, tokens):
        feature = defaultdict(int)
        NG = self.ngram2id
        for t in tokens:
            if (t,) not in NG:
                NG[t,] = len(NG)
            feature[NG[t,]] += 1
        return feature

    def bigram(self, tokens):
        feature = self.unigram(tokens)
        NG = self.ngram2id
        for x, y in zip(tokens[:-1], tokens[1:]):
            if (x, y) not in NG:
                NG[x, y] = len(NG)
            feature[NG[x, y]] += 1
        return feature

    def save(self, save_file):
        self.id2ngram = _dict2list(self.ngram2id)
        config = {'id2ngram': self.id2ngram}
        pickle.dump(config, open(save_file, 'wb'), -1)

    def load(self, load_file):
        config = pickle.load(load_file, 'rb')
        self.id2ngram = config['id2ngram']
        self.ngram2id = _list2dict(self.id2ngram)
        return self


class TractorClassMapping(object):
    def __init__(self):
        self.class2id = {}
        self.id2class = None

    def to_index(self, class_name):
        if class_name in self.class2id:
            return self.class2id[class_name]

        index = len(self.class2id)
        self.class2id[class_name] = index
        return index

    def to_class_name(self, index):
        if self.id2class is None:
            self.id2class = _dict2list(self.class2id)
        if index == -1:
            return "**not in training**"
        if index >= len(self.id2class):
            raise KeyError(
                'class index ({0}) should be less than the number of classes ({1}).'.format(index, len(self.id2class)))
        return self.id2class[index]

    def save(self, save_file):
        self.id2class = _dict2list(self.class2id)
        config = {'id2class': self.id2class}
        pickle.dump(config, open(save_file, 'wb'), -1)

    def load(self, load_file):
        config = pickle.load(open(load_file, 'rb'))
        self.id2class = config['id2class']
        self.class2id = _list2dict(self.id2class)
        return self


class TractorTextSource(object):
    def __init__(self, method, document_tokenize=None):
        self.text_preprocessor = TractorTextPreProcessor()
        self.feature_generator = TractorFeatureGenerator()
        self.class_mapping = TractorClassMapping()
        self.document_tokenize = document_tokenize

    @staticmethod
    def __read_text_source(text_source, delimiter):
        if isinstance(text_source, str):
            with open(text_source, 'r') as f:
                text_source = [line.split(delimiter) for line in f]
        elif not isinstance(text_source, list):
            raise TypeError('text_source should be list or str')
        return text_source

    def get_class_index(self, class_name):
        self.class_mapping.to_index(class_name)

    def get_class_name(self, class_index):
        return self.class_mapping.to_class_name(class_index)

    def to_model(self, document, class_name=None):
        feature = self.feature_generator.bigram(self.text_preprocessor.preprocess(document, self.document_tokenize))
        if class_name is None:
            return feature
        return feature, self.class_mapping.to_index(class_name)

    def text_source(self, text_source, delimiter, feature_path=None):
        if not feature_path:
            feature_path = '%s.svm' % text_source
        text_source = self.__read_text_source(text_source, delimiter)
        with open(feature_path, 'w') as w:
            for line in text_source:
                try:
                    label, text = line
                except ValueError:
                    continue
                feature, label = self.to_model(text, label)
                w.write('%s %s\n' % (label, ''.join(' {0}:{1}'.format(f, feature[f]) for f in sorted(feature))))

    def save(self, save_file):
        config = {
            'text_preprocessor': 'text_preprocessor.config.pickle',
            'feature_generator': 'feature_generator.config.pickle',
            'class_mapping': 'class_mapping.config.pickle',
        }
        if not os.path.exists(save_file):
            os.mkdir(save_file)
        self.text_preprocessor.save(os.path.join(save_file, config['text_preprocessor']))
        self.feature_generator.save(os.path.join(save_file, config['feature_generator']))
        self.class_mapping.save(os.path.join(save_file, config['class_ma']))

    def load(self, load_file):
        config = {
            'text_preprocessor': 'text_preprocessor.config.pickle',
            'feature_generator': 'feature_generator.config.pickle',
            'class_mapping': 'class_mapping.config.pickle',
        }
        self.text_preprocessor.load(os.path.join(load_file, config['text_preprocessor']))
        self.feature_generator.load(os.path.join(load_file, config['feature_generator']))
        self.class_mapping.load(os.path.join(load_file, 'class_mapping'))
        return self


# FEATURE
class TractorFeature(object):
    def __init__(self):
        pass


# PREDICT RESULT
class TractorPredictResult(object):
    def __init__(self):
        pass
