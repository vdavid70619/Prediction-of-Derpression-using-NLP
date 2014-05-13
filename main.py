"""
    Main function using word2vec and clustering
"""

import os
import cPickle as pickle

from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import f1_score

from smote import *
import numpy as np
from csv_dataloader import *
from get_word2vec import *
from get_clusters import *
from get_LIWC import *
from encoder import *
from preprocess import *

## save a class object to a file using pickle
def save(obj, filename):
    with open(filename, 'wb+') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def main(n_fold=10):
    ### Load data
    dataloader = csv_dataloader(extrafile='data/fixed_train_gender_class.csv', extra=True)
    if not os.path.exists('output/data_cache.pk'):
        dataloader.read_csv(applyfun=preprocess)
        dataloader.save('output/data_cache.pk')
    else:
        dataloader.load('output/data_cache.pk')
    dataloader.summary()
    print "Read in finished"

    ### Load pre-train word2vector model
    word2vec = get_word2vec(model='data/GoogleNews-vectors-negative300.bin', binary=True, size=300)
    print 'Pretrained word2vec loaded'

    ### Train BoW
    n_topics = 25
    model_file = 'output/clustering_gmm_25.pk';
    clusters = get_clusters(method='gmm', n_topics=n_topics)
    if not os.path.exists(model_file):
        ### Convert word to vector
        tokens = sum(dataloader.data.viewvalues(), [])
        print '#Tokens from training data: ' + str(len(tokens))
        train_vectors = word2vec.encode(tokens)
        print '#Vectors from training data: ' + str(len(train_vectors))
        print 'Training Clusters...'
        clusters.fit(train_vectors)
        clusters.save(model_file)
        clusters.summary()
    else:
        print 'Cluster Model Loaded...'
        clusters.load(model_file)

    ### Calculate LIWC hist
    LIWC = get_LIWC()
    #print LIWC.calculate_hist(tokens, normalize=False)

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

        encode_pos = encode_feature(train_data, train_pos_id, [word2vec, clusters, LIWC])
        encode_pos = SMOTE(encode_pos, 200, len(train_pos_id)/4)
        label_pos = np.ones(len(encode_pos))

        encode_neg = encode_feature(train_data, train_pos_id, [word2vec, clusters, LIWC])
        label_neg = np.zeros(len(encode_neg))

        encode = np.concatenate((encode_pos, encode_neg), axis=0)
        label = np.concatenate((label_pos, label_neg), axis=0)
        print encode.shape
        print label.shape

        encode = preprocessing.scale(encode)
        classifier = svm.LinearSVC(verbose=True)
        #weight = label+1 # pos:neg = 2:1 for imbalanced training
        classifier.fit(encode, label)
        print 'F1 score: ' + str(f1_score(label, classifier.predict(encode)))


        ### ============================================================
        ###                         Test Part
        ### ============================================================
        print 'Testing>>>>>>>>>>>>>>>>>>>>>>>>>'

        test_data = dataloader.data_retrieve(test_id)
        encode = encode_feature(test_data, test_id, [word2vec, clusters, LIWC])
        label = dataloader.label_retrieve(test_id)

        encode = preprocessing.scale(encode)
        print 'F1 score: ' + str(f1_score(label, classifier.predict(encode)))
        fscores.append(f1_score(label, classifier.predict(encode)))
        models.append(classifier)

    print 'MEAN F1 score: ' + str(np.mean(fscores))
    print 'BEST F1 score: ' + str(np.max(fscores)) + ' by Model ' + str(np.argmax(fscores)+1)
    print 'VAR F1 score: ' + str(np.var(fscores))

    save(models[np.argmax(fscores)], 'output/model_clustering_' + str(fscores[np.argmax(fscores)]) + '.pk')

if __name__ == "__main__":
    main()
