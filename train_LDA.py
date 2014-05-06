"""
    Main function
"""

import re
import os
import nltk
# import cPickle as pickle

from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import f1_score
import numpy as np

from csv_dataloader import *
from get_topics import *


## save a class object to a file using pickle
# def save_object(obj, filename):
#     with open(filename, 'w+') as output:
#         pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def has_pattern(word, pattern):
    pass


def preprocess(words):

    words = unicode(words, errors='ignore') #This is for gensim
    tokens = nltk.WordPunctTokenizer().tokenize(words.lower())
    # tokens = list(set(tokens))
    stopwords = nltk.corpus.stopwords.words('english')
    tokens = [w for w in tokens if w not in stopwords]
    tokens = [w for w in tokens if len(w)<20 and len(w)>2]
    tokens = [w for w in tokens if re.match('\W+',w)==None]
    return tokens

def main():

    n_fold = 5

    dataloader = csv_dataloader(datafile='data/extra_statuses.csv')
    dataloader.read_csv(applyfun=preprocess)
    dataloader.summary()

    ### Get word2id first
    tokens = sum(dataloader.data.viewvalues(), [])
    word2id = get_word2id()
    word2id.fit(tokens)
    ids = word2id.ids()
    word2id.save('word2id.pk')
    print "#Id: " + str(len(ids.keys()))
    print '#Tokens from training data: ' + str(len(tokens))

    ### Train LDA
    n_topics = 100
    topics = get_topics(id2word=ids, method='lda', n_topics=n_topics)
    if not os.path.exists('output/lda_100.pk'):
        print 'Training LDA...'
        topics.fit(tokens)
        topics.save('output/lda_100.pk')
        topics.summary()
    else:
        topics.load('output/lda_100.pk')



if __name__ == "__main__":
    main()
