"""
    Main function using LDA
"""

import os
import cPickle as pickle

import numpy as np
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import f1_score

from csv_dataloader import *
from get_topics import *
from smote import *
from preprocess import *


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
    tokens = sum(dataloader.data.viewvalues(), [])
    word2id = get_word2id()
    if not os.path.exists('output/word2id.pk'):
        word2id.fit(tokens)
        word2id.save('output/word2id.pk')
    else:
        word2id.load('output/word2id.pk')
    ids = word2id.ids()
    print "#Id: " + str(len(ids.keys()))
    print '#Tokens from training data: ' + str(len(tokens))

    ### Train and load LDA
    n_topics = 100
    model_file = 'output/lda_all_100.pk'
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

        train_data, train_ldata, train_label, _, train_gender, train_time = dataloader.batch_retrieve(train_id)

        ### Balance Train Data
        _, train_pos_id, train_neg_id = dataloader.balance(train_id, K=2)

        ### Generate Train Positive and Negative Data Encodings Separately
        encode_pos = np.zeros((len(train_pos_id), n_topics))
        gender_pos = np.zeros((len(train_pos_id), 1))
        time_pos = np.zeros((len(train_pos_id), 4))
        i = 0
        for id in train_pos_id:
            tokens = train_ldata[id]
            #tokens = [train_data[id]]
            encode_pos[i,:] = topics.encode(tokens)
            gender_pos[i] = train_gender[id]
            time_pos[i,:] = [train_time[id].month, train_time[id].day, train_time[id].hour, train_time[id].minute]
            i +=1

        encode_pos = np.concatenate((encode_pos, gender_pos, time_pos), axis=1)

        encode_pos = SMOTE(encode_pos, 200, len(train_pos_id)/4)

        label_pos = np.ones(len(encode_pos))
        encode_neg = np.zeros((len(train_neg_id), n_topics))
        gender_neg = np.zeros((len(train_neg_id), 1))
        time_neg = np.zeros((len(train_neg_id), 4))
        i = 0
        for id in train_neg_id:
            tokens = train_ldata[id]
            #tokens = [train_data[id]]
            encode_neg[i,:] = topics.encode(tokens)
            gender_neg[i] = train_gender[id]
            time_neg[i,:] = [train_time[id].month, train_time[id].day, train_time[id].hour, train_time[id].minute]
            i +=1

        encode_neg = np.concatenate((encode_neg, gender_neg, time_neg), axis=1)
        label_neg = np.zeros(len(encode_neg))

        encode = np.concatenate((encode_pos, encode_neg), axis=0)
        label = np.concatenate((label_pos, label_neg), axis=0)
        print encode.shape
        print label.shape


        encode = preprocessing.scale(encode)
        #classifier = svm.NuSVC(kernel='linear', verbose=True, cache_size=4000)
        classifier = svm.LinearSVC(verbose=True)
        #weight = 10*label+1 # pos:neg = 2:1 for imbalanced training
        classifier.fit(encode, label)
        #print classifier.predict(encode)
        print 'F1 score: ' + str(f1_score(label, classifier.predict(encode)))


        ### ============================================================
        ###                         Test Part
        ### ============================================================
        print 'Testing>>>>>>>>>>>>>>>>>>>>>>>>>'

        test_data, test_ldata, test_label, _, test_gender, test_time = dataloader.batch_retrieve(test_id)

        ### Generate Test Data Encodings
        encode = np.zeros((len(test_id), n_topics))
        label = np.zeros(len(test_id))
        gender = np.zeros((len(test_id),1))
        time = np.zeros((len(test_id),4))
        i = 0
        for id in test_id:
            tokens = test_ldata[id]
            #tokens = [test_data[id]]
            encode[i,:] = topics.encode(tokens)
            label[i] = test_label[id]
            gender[i] = test_gender[id]
            time[i,:] = [test_time[id].month, test_time[id].day, test_time[id].hour, test_time[id].minute]
            i +=1

        encode = np.concatenate((encode, gender, time), axis=1)
        encode = preprocessing.scale(encode)

        print 'F1 score: ' + str(f1_score(label, classifier.predict(encode)))
        fscores.append(f1_score(label, classifier.predict(encode)))
        models.append(classifier)

    print 'MEAN F1 score: ' + str(np.mean(fscores))
    print 'BEST F1 score: ' + str(np.max(fscores)) + ' by Model ' + str(np.argmax(fscores)+1)
    print 'VAR F1 score: ' + str(np.var(fscores))

    save(models[np.argmax(fscores)], 'output/model_LDA_' + str(fscores[np.argmax(fscores)]) + '.pk')

if __name__ == "__main__":
    main()
