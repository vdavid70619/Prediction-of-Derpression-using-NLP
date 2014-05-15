import os
import matplotlib.pyplot as plt
from sklearn import manifold
from nltk import FreqDist

from csv_dataloader import *
from get_topics import *
from get_clusters import *
from get_word2vec import *


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


def draw_word2vec():
    ### Load data
    dataloader = csv_dataloader()
    dataloader.load('output/data_cache.pk')
    print "Read in finished"

    ### Load pre-train word2vector model
    word2vec = get_word2vec(model='data/GoogleNews-vectors-negative300.bin', binary=True, size=300)
    print 'Pretrained word2vec loaded'

    all_tokens = sum(dataloader.data.viewvalues(), [])
    print '#Tokens: ' + str(len(all_tokens))
    fdist = FreqDist(all_tokens)
    tokens = fdist.keys()[1:500]
    print tokens
    tokens_has_vectors = []
    for token in tokens:
        if word2vec[token] is not None:
            tokens_has_vectors.append(token)

    print '#Unique Tokens \w Vectors: ' + str(len(tokens_has_vectors))
    vectors = word2vec.encode(tokens_has_vectors)
    print '#Unique Vectors: ' + str(len(vectors))

    print("Computing MDS embedding")
    clf = manifold.MDS(n_components=2, n_init=1, max_iter=2000)
    #clf = manifold.Isomap(n_components=2, max_iter=100)
    vectors_mds = clf.fit_transform(vectors)
    print("Done. Stress: %f" % clf.stress_)
    plot_embedding(vectors_mds, tokens_has_vectors, "MDS embedding of the words")

if __name__ == '__main__':
    draw_word2vec()