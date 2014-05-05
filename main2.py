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

    dataloader = csv_dataloader()
    dataloader.read_csv(applyfun=preprocess)
    dataloader.summary()

    ### Get word2id first
    tokens = sum(dataloader.data.viewvalues(), [])
    word2id = get_word2id()
    word2id.fit(tokens)
    ids = word2id.ids()
    print "#Id: " + str(len(ids.keys()))


    ### ============================================================
    ###                         n fold
    ### ============================================================

    nfolds = dataloader.nfold(n_fold)
    fscores = []

    for fold_ind in range(n_fold):

        print '======================== FOLD ' + str(fold_ind+1) + '========================'

        test_id = nfolds[fold_ind]
        train_id = []
        for i in range(n_fold):
            if i != fold_ind:
                train_id += nfolds[i]

        ### ============================================================
        ###                         Train Part
        ### ============================================================
        print 'Training>>>>>>>>>>>>>>>>>>>>>>>>>'

        train_data, train_ldata, train_label, train_score = dataloader.batch_retrieve(train_id)

        ### Train LDA
        tokens = sum(train_ldata.viewvalues(), [])
        print '#Tokens from training data: ' + str(len(tokens))

        n_topics = 25
        topics = get_topics(id2word=ids, method='lda', n_topics=n_topics)
        if not os.path.exists('output/lda_25.pk'):
            print 'Training LDA...'
            topics.fit(tokens)
            topics.save('output/lda_25.pk')
            topics.summary()
        else:
            topics.load('output/lda_25.pk')

        ### Balance Train Data
        train_id = dataloader.balance(train_id)

        ### Generate Train Data Encodings
        encode = np.zeros((len(train_id), n_topics))
        label = np.zeros(len(train_id))
        score = np.zeros(len(train_id))
        i = 0
        for id in train_id:
            tokens = train_ldata[id]
            encode[i,:] = topics.encode(tokens)
            label[i] = train_label[id]
            # score[i] = train_score[id]
            i +=1

        #print encode
        #print label

        encode = preprocessing.scale(encode)

        classifier = svm.NuSVC(nu=0.5, kernel='linear', verbose=True, cache_size=4000)
        weight = label+1 # pos:neg = 2:1 for imbalanced training
        classifier.fit(encode, label, weight)

        #print classifier.predict(encode)
        print 'F1 score: ' + str(f1_score(label, classifier.predict(encode)))


        ### ============================================================
        ###                         Test Part
        ### ============================================================
        print 'Testing>>>>>>>>>>>>>>>>>>>>>>>>>'

        test_data, test_ldata, test_label, _ = dataloader.batch_retrieve(test_id)

        ### Generate Test Data Encodings
        encode = np.zeros((len(test_id), n_topics))
        label = np.zeros(len(test_id))
        i = 0
        for id in test_id:
            tokens = test_ldata[id]
            encode[i,:] = topics.encode(tokens)
            label[i] = test_label[id]
            i +=1

        #print encode
        #print label
        #print classifier.predict(encode)

        encode = preprocessing.scale(encode)

        print 'F1 score: ' + str(f1_score(label, classifier.predict(encode)))
        fscores.append(f1_score(label, classifier.predict(encode)))

    print 'MEAN F1 score: ' + str(np.mean(fscores))
    print 'VAR F1 score: ' + str(np.var(fscores))

if __name__ == "__main__":
    main()
