"""
    Train Word Frequency using Lasso
"""

import os

from csv_dataloader import *
from nltk import FreqDist

def normalize(data,base):
    for word in base:
        if word not in data:
            #print word
            data[word] = 0
    #print len(data)
    return dict(sorted(data.items(), key=lambda x: x[0]))


def train_wordfrequency(n_dims = 50):
    ### Load data
    dataloader = csv_dataloader()
    dataloader.load('output/data_cache.pk')
    print "Read in finished"

    train_id = dataloader.id
    _, pos_id, neg_id = dataloader.balance(train_id, 'full')
    train_data_pos = dataloader.data_retrieve(pos_id)
    train_data_neg = dataloader.data_retrieve(neg_id)
    tokens = sum(dataloader.data.viewvalues(), [])
    tokens_pos = sum(train_data_pos['data'].viewvalues(), [])
    tokens_neg = sum(train_data_neg['data'].viewvalues(), [])

    fdist_base = FreqDist(tokens)

    fdist_pos = FreqDist(tokens_pos)
    fdist_pos = normalize(fdist_pos, fdist_base)
    fdist_neg = FreqDist(tokens_neg)
    fdist_neg = normalize(fdist_neg, fdist_base)

    print list(fdist_pos.viewkeys())[:100]
    print list(fdist_neg.viewkeys())[:100]

    labels_pos = [1] * len(tokens_pos)
    labels_neg = [0] * len(tokens_neg)

    labels = labels_pos + labels_neg
    corpus = tokens_pos + tokens_neg

if __name__ == '__main__':
    train_wordfrequency()