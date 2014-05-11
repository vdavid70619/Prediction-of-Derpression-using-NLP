"""
    Train LDA using extra data
"""

import os
from csv_dataloader import *
from get_topics import *
from preprocess import *

def train_LDA(n_topics=100):

    ### Load extra data
    dataloader = csv_dataloader(datafile='data/extra_statuses.csv')
    if not os.path.exists('output/extra_cache.pk'):
        dataloader.read_csv(applyfun=preprocess, verbose=True)
        dataloader.save('output/extra_cache.pk')
    else:
        dataloader.load('output/extra_cache.pk')
    tokens = sum(dataloader.ldata.viewvalues(), [])
    print '#Tokens from training data: ' + str(len(tokens))
    print 'Readin done'

    ### Get word2id first
    word2id = get_word2id()
    if not os.path.exists('word2id.pk'):
        word2id.fit(tokens)
        word2id.save('word2id.pk')
    else:
        word2id.load('word2id.pk')
    ids = word2id.ids()
    print "#Id: " + str(len(ids.keys()))

    ### Train LDA
    topics = get_topics(id2word=ids, method='lda', n_topics=n_topics)
    if not os.path.exists('output/lda_all_'+str(n_topics)+'.pk'):
        print 'Training LDA...'
        topics.fit(tokens)
        topics.save('output/lda_all_'+str(n_topics)+'.pk')
        topics.summary()
    else:
        topics.load('output/lda_all_'+str(n_topics)+'.pk')


def show_topics(n_topics=100):
    ### Get word2id first
    word2id = get_word2id()
    word2id.load('word2id.pk')
    ids = word2id.ids()
    print "#Id: " + str(len(ids.keys()))

    ### Show LDA
    topics = get_topics(id2word=ids, method='lda', n_topics=n_topics)
    topics.load('output/lda_all_100.pk')
    topics.summary()

if __name__ == "__main__":
    # train_LDA()
    show_topics()