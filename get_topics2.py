'''
    Topic modeling method to learn the topic and generate sparse code
    Xiyang
'''

from LLDA import *
import numpy as np

import cPickle as pickle

class get_topics2(object):
    def __init__(self, n_topics=2, alpha=0.001, beta=0.001, max_iter=100, method='LLDA'):
        self.method = method.lower()
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.iteration = max_iter
        assert self.method=='llda', 'Only Labeled LDA method is implemented in thi library'

        self.model = LLDA(self.n_topics, self.alpha, self.beta)


    def fit(self, X, y=None, verbose=False):
        '''
        X: Gensim corpora form
        # Gensim treat corpus input as chunks of lists
        '''

        assert isinstance(X[0], list), 'Not chuncks of lists for docs. Require [[doc1],[doc2],...] as gensim chunk format'
        if y is not None:
            assert isinstance(y[0], list), 'Not chuncks of lists for labels. Require [[label1],[label2],...] as gensim chunk format'

        self.labelset = list(set(reduce(list.__add__, y)))
        self.model.set_corpus(self.labelset, X, y)

        for i in range(self.iteration):
            if verbose:
                print  "-- %d : %.4f" % (i, self.model.perplexity())
            self.model.inference()

        if verbose:
            print "Final perplexity : %.4f" % self.model.perplexity()


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
        Result is normalized any way.
        '''

        assert isinstance(X[0], list), 'Not chuncks of lists. Require [[],[],...] as gensim chunk format'

        hist = self.model.perplexity(docs=X)
        sort_ind = np.argsort(hist)[::-1]  # reverse index sequence after argsort
        hist[sort_ind[topk:]] = 0
        return hist


    def summary(self, n_topics=-1):
        phi = self.model.phi()
        for k in range(np.size(phi, axis=0)):
            print "\n%d: " % k,
            for w in numpy.argsort(-phi[k])[:20]:
                print "+ %.4f*%s" % (phi[k,w], self.model.vocas[w]),


