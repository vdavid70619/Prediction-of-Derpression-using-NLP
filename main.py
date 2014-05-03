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


from csv_dataloader import *
from get_word2vec import *
from get_clusters import *


## save a class object to a file using pickle
# def save_object(obj, filename):
#     with open(filename, 'w+') as output:
#         pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def has_pattern(word, pattern):
    pass


def preprocess(words):

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
    dataloader.read_csv()
    dataloader.summary()

    ### ============================================================
    ###                         n fold
    ### ============================================================


    ### Load pre-train word2vector model
    word2vec = get_word2vec(model='data/GoogleNews-vectors-negative300.bin', binary=True, size=300)

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

        train_data, train_label, train_score = dataloader.batch_retrieve(train_id)

        ### Train BoW
        words = str(train_data.viewvalues())
        tokens = preprocess(words)
        print '#Tokens from training data: ' + str(len(tokens))

        ### Convert word to vector
        train_vectors = word2vec.batch_convert(tokens)
        print '#Vectors from training data: ' + str(len(train_vectors))

        n_topics = 256
        clusters = get_clusters(method='kmeans', n_topics=n_topics)
        if not os.path.exists('output/clustering_256.pk'):
            print 'Training Clusters...'
            clusters.fit(train_vectors)
            clusters.save('output/clustering_256.pk')
            clusters.summary()
        else:
            clusters.load('output/clustering_256.pk')

        train_id = dataloader.balance(train_id)

        ### Generate Train Data Encodings
        encode = zeros((len(train_id), n_topics))
        label = zeros(len(train_id))
        score = zeros(len(train_id))
        i = 0
        for id in train_id:
            tokens = preprocess(train_data[id])
            vec = word2vec.batch_convert(tokens)
            encode[i,:] = clusters.encode(vec)
            label[i] = train_label[id]
            # score[i] = train_score[id]
            i +=1

        #print encode
        #print label

        encode = preprocessing.scale(encode)

        classifier = svm.NuSVC(nu=0.5, kernel='linear', verbose=True)
        #weight = label+1 # pos:neg = 2:1 for imbalanced training
        classifier.fit(encode, label)

        #print classifier.predict(encode)
        print 'F1 score: ' + str(f1_score(label, classifier.predict(encode)))


        ### ============================================================
        ###                         Test Part
        ### ============================================================
        print 'Testing>>>>>>>>>>>>>>>>>>>>>>>>>'

        test_data, test_label, _ = dataloader.batch_retrieve(test_id)

        ### Generate Test Data Encodings
        encode = zeros((len(test_id), n_topics))
        label = zeros(len(test_id))
        i = 0
        for id in test_id:
            tokens = preprocess(test_data[id])
            vec = word2vec.batch_convert(tokens)
            encode[i,:] = clusters.encode(vec)
            label[i] = test_label[id]
            i +=1

        #print encode
        #print label
        #print classifier.predict(encode)

        encode = preprocessing.scale(encode)

        print 'F1 score: ' + str(f1_score(label, classifier.predict(encode)))
        fscores.append(f1_score(label, classifier.predict(encode)))

    print 'MEAN F1 score: ' + str(mean(fscores))

if __name__ == "__main__":
    main()
