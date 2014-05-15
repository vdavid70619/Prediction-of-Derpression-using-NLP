"""
    Train Labeled LDA using train data
"""

import os
from nltk import FreqDist

from csv_dataloader import *
from get_topics2 import *
def score2label(score):
    if score in range(0,15):
        return 'level1'
    elif score in range(15,20):
        return 'level2'
    elif score in range(20,25):
        return 'level3'
    elif score in range(25,30):
        return 'level4'
    elif score in range(30,34):
        return 'level5'
    elif score in range(34,40):
        return 'level6'
    else:
        return 'level7'

def train_LLDA(n_topics=2):
    ### Load data
    dataloader = csv_dataloader()
    dataloader.load('output/data_cache.pk')
    print "Read in finished"

    train_id = dataloader.id
    train_data = dataloader.data_retrieve(train_id)

    labels = []
    corpus = []

    for id in train_id:
        corpus.append(train_data['ldata'][id])
        labels.append(score2label(train_data['score'][id]))

    ### Train LLDA
    topics = get_topics2(method='LLDA', max_iter=2)
    model_file = 'output/LLDA_8.pk'
    if not os.path.exists(model_file):
        print 'Training LLDA...'
        topics.fit(corpus, labels, verbose=True)
        topics.save(model_file)
    else:
        topics.load(model_file)
    topics.summary()


if __name__ == '__main__':
    train_LLDA()