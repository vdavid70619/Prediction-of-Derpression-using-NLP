'''
    Topic modeling method to learn the topic and generate sparse code
    Xiyang
'''

from gensim import models
from gensim import corpora
import numpy as np

import cPickle as pickle

class get_word2id(object):
    def __init__(self):
        self._id = corpora.Dictionary()

    def fit(self, X):
        self._id = corpora.Dictionary([X])

    def ids(self):
        return self._id

    def load(self, filename):
        with open(filename, 'rb') as input:
            self._id = pickle.load(input)

    def save(self, filename):
        with open(filename, 'wb+') as output:
            ## save a class object to a file using pickle
            pickle.dump(self._id, output, pickle.HIGHEST_PROTOCOL)

class get_topics(object):
    def __init__(self, id2word, method='LDA', n_topics=20):
        self.method = method.lower()
        self.n_topics = n_topics
        self.id2word = id2word
        self.model = None

        if self.method == 'hdp':
            self.model = models.HdpModel(id2word=self.id2word)
        else:
            self.model = models.LdaModel(id2word=self.id2word, num_topics=self.n_topics)


    def fit(self, X):
        '''
        X: Gensim corpora form
        # Gensim treat corpus input as chunks of lists
        '''

        assert isinstance(X[0], list), 'Not chuncks of lists. Require [[],[],...] as gensim chunk format'

        # Generate the bag of word corpus.
        mm = [self.id2word.doc2bow(line) for line in X]

        if self.method == 'hdp':
            self.model = models.HdpModel(corpus=mm, id2word=self.id2word)
        else:
            self.model = models.LdaModel(corpus=mm, id2word=self.id2word, num_topics=self.n_topics)

    def load(self, filename):
        with open(filename, 'rb') as input:
            self.model = pickle.load(input)

    def save(self, filename):
        with open(filename, 'wb+') as output:
            ## save a class object to a file using pickle
            pickle.dump(self.model, output, pickle.HIGHEST_PROTOCOL)

    def encode(self, X, topk=20, normalize=True):
        '''
        # Gensim treat corpus input as chunks of lists
        # First one in the return tupleis chunks of gammas.
        We have only one trunk
        '''

        assert isinstance(X[0], list), 'Not chuncks of lists. Require [[],[],...] as gensim chunk format'

        mm = [self.id2word.doc2bow(line) for line in X]
        # Update the LDA model
        #self.model.update(corpus=mm)

        gammas = self.model.inference(mm)[0]
        hist = np.sum(gammas, axis=0)
        if normalize:
            hist = hist/np.sum(hist, axis=0)
        #print hist
        sort_ind = np.argsort(hist)[::-1]  # reverse index sequence after argsort
        hist[sort_ind[topk:]]=0
        return hist


    def summary(self):
        print '#Topics: ' + str(self.model.num_topics)

        if self.method == 'lda':
            topics = self.model.show_topics()
        elif self.method == 'hdp':
            topics = self.model.print_topics(topics=self.n_topics, topn=10)

        i=1
        for topic in topics:
            print str(i) + ': ' + topic
            i += 1


