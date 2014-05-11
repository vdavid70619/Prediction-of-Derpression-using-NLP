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
def save(obj, filename):
    with open(filename, 'wb+') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


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

    n_fold = 10

    dataloader = csv_dataloader(extrafile='data/fixed_train_gender_class.csv', extra=True)
    if not os.path.exists('output/data_cache.pk'):
        dataloader.read_csv(applyfun=preprocess)
        dataloader.save('output/data_cache.pk')
    else:
        dataloader.load('output/data_cache.pk')
    dataloader.summary()
    print "Read in finished"

    ### ============================================================
    ###                         n fold
    ### ============================================================


    ### Load pre-train word2vector model
    word2vec = get_word2vec(model='data/GoogleNews-vectors-negative300.bin', binary=True, size=300)
    print 'Pretrained word2vec loaded'

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

        train_data, _, train_label, train_score, _, _ = dataloader.batch_retrieve(train_id)

        ### Train BoW
        tokens = str(train_data.viewvalues())
        print '#Tokens from training data: ' + str(len(tokens))

        ### Convert word to vector
        train_vectors = word2vec.batch_convert(tokens)
        print '#Vectors from training data: ' + str(len(train_vectors))

        n_topics = 25
        clusters = get_clusters(method='gmm', n_topics=n_topics)
        if not os.path.exists('output/clustering_gmm_25.pk'):
            print 'Training Clusters...'
            clusters.fit(train_vectors)
            clusters.save('output/clustering_gmm_25.pk')
            clusters.summary()
        else:
            print 'Cluster Model Loaded...'
            clusters.load('output/clustering_gmm_25.pk')

        ### Balance Train Data
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
        weight = label+1 # pos:neg = 2:1 for imbalanced training
        classifier.fit(encode, label, weight)

        #print classifier.predict(encode)
        print 'F1 score: ' + str(f1_score(label, classifier.predict(encode)))


        ### ============================================================
        ###                         Test Part
        ### ============================================================
        print 'Testing>>>>>>>>>>>>>>>>>>>>>>>>>'

        test_data, _, test_label, _, _, _ = dataloader.batch_retrieve(test_id)

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
        models.append(classifier)

    print 'MEAN F1 score: ' + str(np.mean(fscores))
    print 'BEST F1 score: ' + str(np.max(fscores)) + ' by Model ' + str(np.argmax(fscores)+1)
    print 'VAR F1 score: ' + str(np.var(fscores))

    save(models[np.argmax(fscores)], 'output/model_' + str(fscores[np.argmax(fscores)]) + '.pk')

if __name__ == "__main__":
    main()
