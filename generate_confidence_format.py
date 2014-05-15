import cPickle as pickle
import re
import os
import nltk

from sklearn import preprocessing
from sklearn import svm

from csv_dataloader import *
from get_topics import *
from get_topics import *
from preprocess import *
from get_word2vec import *
from get_clusters import *
from get_LIWC import *
from encoder1 import *

def load(filename):
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
        return obj


def predict():

    dataloader = csv_dataloader(datafile='data/test_statuses.csv', extrafile='data/test_metadata.csv', nolabel=True, extra=True)
    if not os.path.exists('output/test_cache.pk'):
        dataloader.read_csv(applyfun=preprocess)
        dataloader.save('output/test_cache.pk')
    else:
        dataloader.load('output/test_cache.pk')
    dataloader.summary()
    print "Read in finished"

    word2id = get_word2id()
    word2id.load('output/word2id.pk')
    ids = word2id.ids()

    n_topics = 100
    topics = get_topics(id2word=ids, method='lda', n_topics=n_topics)
    topics.load('output/lda_all_100.pk')

    # ### Load pre-train word2vector model
    # word2vec = get_word2vec(model='data/GoogleNews-vectors-negative300.bin', binary=True, size=300)
    # print 'Pretrained word2vec loaded'
    #
    # n_topics = 100
    # model_file = 'output/clustering_gmm_100.pk';
    # clusters = get_clusters(method='gmm', n_topics=n_topics)
    # print 'Cluster Model Loaded...'
    # clusters.load(model_file)

    ### Calculate LIWC hist
    LIWC = get_LIWC()

    test_id = dataloader.id
    test_data = dataloader.data_retrieve(test_id)

    ## Generate Test Data Encodings
    encode = encode_feature(test_data, test_id, [topics, LIWC])

    print encode

    encode = preprocessing.scale(encode)
    classifier = load('output/model_LDA_100_0.461538461538.pk')
    predict_label = classifier.predict(encode)
    predict_prob = classifier.predict_proba(encode)

    with open('output/result.csv', 'w+') as file:
        file.write('userID, binaryPrediction, confidence, regression\n')

        for i in range(len(predict_label)):
            string = test_id[i] + ', '
            if predict_label[i]==1:
                string += '+, '
            else:
                string += '-, '
            string += str(predict_prob[i][1]) + ', '
            string += 'N\n'
            file.write(string)

    print str(sum(predict_label)) + '/' + str(len(predict_label))




if __name__ == "__main__":
    predict()