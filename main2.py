"""
    Main function using LDA
"""

import os
import cPickle as pickle

import numpy as np
from sklearn import preprocessing
from sklearn import svm
from sklearn import ensemble
from sklearn.metrics import f1_score
from smote import *

from csv_dataloader import *
from get_topics import *
from get_LIWC import *
from preprocess import *
from encoder1 import *

## save a class object to a file using pickle
def save(obj, filename):
    with open(filename, 'wb+') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def main(n_fold=10):
    ### Load trainning data
    dataloader = csv_dataloader(extrafile='data/fixed_train_gender_class.csv', extra=True)
    if not os.path.exists('output/data_cache.pk'):
        dataloader.read_csv(applyfun=preprocess)
        dataloader.save('output/data_cache.pk')
    else:
        dataloader.load('output/data_cache.pk')
    dataloader.summary()
    print "Read in finished"

    ### Get word2id first
    tokens = sum(dataloader.ldata.viewvalues(), [])
    word2id = get_word2id()
    if not os.path.exists('output/word2id.pk'):
        word2id.fit(tokens)
        word2id.save('output/word2id.pk')
    else:
        word2id.load('output/word2id.pk')
    ids = word2id.ids()
    print "#Id: " + str(len(ids.keys()))
    print '#Tokens from training data: ' + str(len(tokens))

    ### Calculate LIWC hist
    LIWC = get_LIWC()
    #print LIWC.calculate_hist(tokens, normalize=False)

    ### Train and load LDA
    n_topics = 25
    model_file = 'output/lda_all_25    .pk'
    topics = get_topics(id2word=ids, method='lda', n_topics=n_topics)
    if not os.path.exists(model_file):
        print 'Training LDA...'
        topics.fit(tokens)
        topics.save(model_file)
        topics.summary()
    else:
        topics.load(model_file)


    ### ============================================================
    ###                         n fold
    ### ============================================================

    nfolds = dataloader.nfold(n_fold)
    fscores = []
    models = []

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

        train_data = dataloader.data_retrieve(train_id)

        ### Balance Train Data
        _, train_pos_id, train_neg_id = dataloader.balance(train_id, K=2)

        encode_pos = encode_feature(train_data, train_pos_id, [topics, LIWC])
        encode_pos = SMOTE(encode_pos, 200, len(train_pos_id)/4)
        label_pos = np.ones(len(encode_pos))

        encode_neg = encode_feature(train_data, train_pos_id, [topics, LIWC])
        label_neg = np.zeros(len(encode_neg))

        encode = np.concatenate((encode_pos, encode_neg), axis=0)
        label = np.concatenate((label_pos, label_neg), axis=0)
        print encode.shape
        print label.shape

        ### Train
        encode = preprocessing.scale(encode)
        classifier = svm.LinearSVC(verbose=True)
        classifier.fit(encode, label)
        print 'F1 score: ' + str(f1_score(label, classifier.predict(encode)))


        ### ============================================================
        ###                         Test Part
        ### ============================================================
        print 'Testing>>>>>>>>>>>>>>>>>>>>>>>>>'

        test_data = dataloader.data_retrieve(test_id)

        ### Generate Test Data Encodings
        encode = encode_feature(test_data, test_id, [topics, LIWC])
        label = dataloader.label_retrieve(test_id)

        ### Test
        encode = preprocessing.scale(encode)
        print 'F1 score: ' + str(f1_score(label, classifier.predict(encode)))

        fscores.append(f1_score(label, classifier.predict(encode)))
        models.append(classifier)

    print 'MEAN F1 score: ' + str(np.mean(fscores))
    print 'BEST F1 score: ' + str(np.max(fscores)) + ' by Model ' + str(np.argmax(fscores)+1)
    print 'VAR F1 score: ' + str(np.var(fscores))

    save(models[np.argmax(fscores)], 'output/model_LDA_' + str(n_topics) + '_' + str(fscores[np.argmax(fscores)]) + '.pk')

if __name__ == "__main__":
    main()
