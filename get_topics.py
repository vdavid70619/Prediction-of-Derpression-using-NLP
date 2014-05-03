'''
    Topic modeling method to learn the topic and generate sparse code
    Xiyang
'''

from gensim import models
from numpy import *

import cPickle as pickle


class get_topics(object):
    def __init__(self, method='LDA', n_topics=20):
        self.method = method.lower()
        self.n_topics = n_topics

    def fit(self, X):
        if self.method == 'hdp':
            self.model = models.HdpModel(X)
        else:
            self.model = models.LdaModel(X, num_topics=self.n_topics)

    def load(self, filename):
        self.model.load(filename)

    def save(self, filename):
        self.model.save(filename)

    def encode(self, X, topk=20):
        gammas = self.model.inference(X)
        sort_ind = argsort(hist)[::-1]  # reverse index sequence after argsort
        gammas[sort_ind[topk:]]=0
        return gammas

    def summary(self):
        if self.method == 'lda':
            print self.model.show_topics()
        elif self.method == 'hdp':
            print self.model.print_topics(topics=self.n_topics, topn=10)