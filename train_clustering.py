"""
    Train Clustering using extra data
"""

import os
from csv_dataloader import *
from get_clusters import *
from get_word2vec import *
from preprocess import *

def train_clustering(n_topics=100, method='gmm'):

    ### Load extra data
    dataloader = csv_dataloader(datafile='data/extra_statuses.csv')
    CACHE_FILE = 'output/data_cache.pk'
    if not os.path.exists(CACHE_FILE):
        dataloader.read_csv(applyfun=preprocess, verbose=True)
        dataloader.save(CACHE_FILE)
    else:
        dataloader.load(CACHE_FILE)
    dataloader.summary()
    tokens = sum(dataloader.data.viewvalues(), [])
    print '#Tokens from training data: ' + str(len(tokens))
    print 'Readin done'

    ### Load pre-train word2vector model
    word2vec = get_word2vec(model='data/GoogleNews-vectors-negative300.bin', binary=True, size=300)
    print 'Pretrained word2vec loaded'
    ### Convert word to vector

    train_vectors = word2vec.batch_convert(tokens)
    print '#Vectors from training data: ' + str(len(train_vectors))

    ### Train Clustering
    clusters = get_clusters(method=method, n_topics=n_topics)
    if not os.path.exists('output/clustering_'+ method + '_' + str(n_topics) + '.pk'):
        print 'Training Clusters...'
        clusters.fit(train_vectors)
        clusters.save('output/clustering_'+ method + '_' + str(n_topics) + '.pk')
        clusters.summary()
    else:
        print 'Cluster Model Loaded...'
        clusters.load('output/clustering_'+ method + '_' + str(n_topics) + '.pk')


def show_topics(top_n=10):
    pass

if __name__ == "__main__":
    train_clustering()