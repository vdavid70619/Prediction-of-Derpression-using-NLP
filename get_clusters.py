'''
    Clustering method to learn the topic and generate sparse code
    Xiyang
'''

from sklearn.cluster import KMeans
from sklearn.mixture import GMM
import numpy as np

import cPickle as pickle


class get_clusters(object):
    def __init__(self, method='kmeans', n_topics=30):
        self.method = method

        if method.lower() == 'kmeans':
            self.clusters = KMeans(init='k-means++', n_clusters=n_topics, verbose=1, n_jobs=1)
        elif method.lower() == 'gmm':
            self.clusters = GMM(n_components=n_topics)

    def fit(self, X):
        self.clusters.fit(X)

    def load(self, filename):
        with open(filename, 'rb') as input:
            self.clusters = pickle.load(input)

    def save(self, filename):
        with open(filename, 'wb+') as output:
            ## save a class object to a file using pickle
            pickle.dump(self.clusters, output, pickle.HIGHEST_PROTOCOL)

    def encode(self, X, topk=20):
        if self.method == 'kmeans':
            distance = self.clusters.transform(X)
            hist = np.sum(1/(distance+1e-6), axis=0)
        elif self.method == 'gmm':
            probability = self.clusters.predict_proba(X)
            hist = np.sum(probability, axis=0)

        hist = hist/(np.sum(hist, axis=0)+1e-6)
        sort_ind = np.argsort(hist)[::-1]  # reverse index sequence after argsort
        hist[sort_ind[topk:]]=0
        return hist

    def fisher_vector(self, X):
        pass

    def VLAD(self, X):
        pass

    def summary(self):
        print 'Clustering Method: ' + self.method
        print self.clusters.get_params()
