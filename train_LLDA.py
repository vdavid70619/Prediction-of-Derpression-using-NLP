"""
    Train Labeled LDA using extra data
"""

import os
from nltk import FreqDist

from csv_dataloader import *
from get_topics2 import *

def train_LLDA(n_topics=2):
    ### Load data
    dataloader = csv_dataloader()
    dataloader.load('output/data_cache.pk')
    print "Read in finished"

    train_id = dataloader.id
    _, pos_id, neg_id = dataloader.balance(train_id, 'full')
    train_data_pos = dataloader.data_retrieve(pos_id)
    train_data_neg = dataloader.data_retrieve(neg_id)

    tokens_pos = sum(train_data_pos['ldata'].viewvalues(), [])
    tokens_neg = sum(train_data_neg['ldata'].viewvalues(), [])

    labels_pos = [['pos']] * len(tokens_pos)
    labels_neg = [['neg']] * len(tokens_neg)

    labels = labels_pos + labels_neg
    corpus = tokens_pos + tokens_neg

    ### Train LLDA
    topics = get_topics2(method='LLDA', max_iter=2)
    if not os.path.exists('output/LLDA_2.pk'):
        print 'Training LLDA...'
        topics.fit(corpus, labels, verbose=True)
        topics.save('output/LLDA_2.pk')
    else:
        topics.load('output/LLDA_2.pk')
    topics.summary()


if __name__ == '__main__':
    train_LLDA()