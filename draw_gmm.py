import os
import matplotlib.pyplot as plt
from sklearn import manifold
import numpy as np
from sklearn.neighbors import NearestNeighbors

from csv_dataloader import *
from get_clusters import *
from get_word2vec import *

## save a class object to a file using pickle
def save(obj, filename):
    with open(filename, 'wb+') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def plot_embedding(X, tokens, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(tokens[i]),
                fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

    plt.show()


def draw_gmm():
    ### Load data
    dataloader = csv_dataloader()
    dataloader.load('output/data_cache.pk')
    print "Read in finished"

    ### Load pre-train word2vector model
    word2vec = get_word2vec(model='data/GoogleNews-vectors-negative300.bin', binary=True, size=300)
    print 'Pretrained word2vec loaded'

    ### Reverse engineering, build vector2word dictionary
    # vec2word = []
    # words =[]
    # for voc, obj in word2vec.model.vocab:
    #     words.append(voc)
    #     vec2word.append(word2vec.model.syn0[obj.index])

    all_vectors = word2vec.model.syn0


    ### Train BoW
    n_topics = 25
    model_file = 'output/clustering_gmm_100.pk';
    clusters = get_clusters(method='gmm', n_topics=n_topics)

    clusters.load(model_file)
    print clusters.clusters.means_

    knn = NearestNeighbors(n_neighbors=10)
    knn.fit(all_vectors)
    save(knn, 'output\draw_gmm_knn.pk')
    nns =  knn.kneighbors(clusters.clusters.means_, return_distance=False)
    for i in range(np.shape(nns)[0]):
        print 'Topic ' + str(i+1) + ': ',
        for j in range(np.shape(nns)[1]):
            print str(word2vec.model.index2word[nns[i,j]]) + ' ',
        print ''



if __name__ == '__main__':
    draw_gmm()