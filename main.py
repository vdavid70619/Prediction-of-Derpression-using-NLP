"""
    Main function
"""

import re
import os
import nltk
import cPickle as pickle

from numpy import *
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans


from csv_dataloader import *
from get_word2vec import *


## save a class object to a file using pickle
def save_object(obj, filename):
    with open(filename, 'w+') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


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

    nfolds = dataloader.nfold(n_fold)

    train_id = []
    for i in range(n_fold-2):
        train_id += nfolds[i]

    test_id = nfolds[n_fold-1]

    ### ============================================================
    ###                         Train Part
    ### ============================================================
    print 'Training>>>>>>>>>>>>>>>>>>>>>>>>>'

    train_data, train_label, train_score = dataloader.batch_retrieve(train_id)

    ### Train BoW
    words = str(train_data.viewvalues())
    tokens = preprocess(words)
    print '#Tokens from training data: ' + str(len(tokens))

    ### Word to vector
    word2vec = get_word2vec(model='data/GoogleNews-vectors-negative300.bin', binary=True, size=300)
    train_vectors = word2vec.batch_convert(tokens)
    print '#Vectors from training data: ' + str(len(train_vectors))

    n_words = 500
    if not os.path.exists('output/clustering.pk'):
        kmeans = KMeans(init='k-means++', n_clusters=n_words, verbose=1, n_jobs=1)
        kmeans.fit(train_vectors)
        save_object(kmeans, 'output/clustering.pk')
        print kmeans.get_params()
    else:
        kmeans = pickle.load(open('output/clustering.pk', 'r'))

    ### Generate Train Data Encodings
    top_k = 20
    encode = zeros((len(train_id), n_words))
    label = zeros(len(train_id))
    score = zeros(len(train_id))
    i = 0
    for id in train_id:
        tokens = preprocess(train_data[id])
        vec = word2vec.batch_convert(tokens)
        distance = kmeans.transform(vec)
        hist = sum(1/(distance+1e-6), axis=0)
        hist = hist/(sum(hist, axis=0)+1e-6)
        sort_ind = argsort(hist)[::-1]  # reverse index sequence after argsort
        hist[sort_ind[top_k:]]=0
        encode[i,:] = hist
        label[i] = train_label[id]
        # score[i] = train_score[id]
        i +=1

    print encode
    print label

    classifier = svm.NuSVC(kernel='linear', verbose=True)
    weight = label+1 # pos:neg = 2:1 for imbalanced training
    classifier.fit(encode, label, weight)

    print classifier.predict(encode)
    print 'F1 score: ' + str(f1_score(label, classifier.predict(encode)))


    ### ============================================================
    ###                         Test Part
    ### ============================================================
    print 'Testing>>>>>>>>>>>>>>>>>>>>>>>>>'

    test_data, test_label, _ = dataloader.batch_retrieve(test_id)

    ### Generate Test Data Encodings
    encode = zeros((len(test_id), n_words))
    label = zeros(len(test_id))
    i = 0
    for id in test_id:
        tokens = preprocess(test_data[id])
        vec = word2vec.batch_convert(tokens)
        distance = kmeans.transform(vec)
        hist = sum(1/(distance+1e-6), axis=0)
        hist = hist/(sum(hist, axis=0)+1e-6)
        sort_ind = argsort(hist)[::-1]  # reverse index sequence after argsort
        hist[sort_ind[top_k:]]=0
        encode[i,:] = hist
        label[i] = test_label[id]
        i +=1

    print encode
    print label
    print classifier.predict(encode)

    print 'F1 score: ' + str(f1_score(label, classifier.predict(encode)))

if __name__ == "__main__":
    main()
