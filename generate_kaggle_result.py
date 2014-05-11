import cPickle as pickle
import re
import os
import nltk

from sklearn import preprocessing
from sklearn import svm

from csv_dataloader import *
from get_topics import *
from get_topics import *


def load(filename):
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
        return obj

def preprocess(words):

    words = unicode(words, errors='ignore') #This is for gensim
    tokens = nltk.WordPunctTokenizer().tokenize(words.lower())
    # tokens = list(set(tokens))
    stopwords = nltk.corpus.stopwords.words('english')
    tokens = [w for w in tokens if w not in stopwords]
    tokens = [w for w in tokens if len(w)<20 and len(w)>2]
    tokens = [w for w in tokens if re.match('\W+',w)==None]
    return tokens


def predict():

    dataloader = csv_dataloader(datafile='data/test_statuses.csv', extrafile='data/test_metadata.csv', nolabel=True, extra=True)
    if not os.path.exists('output/test_cache.pk'):
        dataloader.read_csv(applyfun=preprocess)
        dataloader.save('output/test_cache.pk')
    else:
        dataloader.load('output/test_cache.pk')
    dataloader.summary()
    print "Read in finished"

    test_id = dataloader.id
    test_data, test_ldata, _, _, test_gender, test_time = dataloader.batch_retrieve(test_id)

    word2id = get_word2id()
    word2id.load('output/word2id.pk')
    ids = word2id.ids()

    n_topics = 100
    topics = get_topics(id2word=ids, method='lda', n_topics=n_topics)
    topics.load('output/lda_all_100.pk')

    ### Generate Test Data Encodings
    encode = np.zeros((len(test_id), n_topics))
    gender = np.zeros((len(test_id),1))
    time = np.zeros((len(test_id),4))
    i = 0
    for id in test_id:
        tokens = test_ldata[id]
        #tokens = [test_data[id]]
        encode[i,:] = topics.encode(tokens)
        gender[i] = test_gender[id]
        time[i,:] = [test_time[id].month, test_time[id].day, test_time[id].hour, test_time[id].minute]
        i +=1

    encode = preprocessing.scale(encode)
    time = preprocessing.scale(time)
    encode = np.concatenate((encode, gender, time), axis=1)

    print encode

    classifier = load('output/model_0.48275862069.pk')
    predict_label = classifier.predict(encode)

    with open('output/result.csv', 'w+') as file:
        file.write('class,predictions\n')
        string = '+,'
        for i in range(len(predict_label)):
            if predict_label[i]==1:
                string += (' ' + test_id[i])

        string += '\n'
        file.write(string)

    print str(sum(predict_label)) + '/' + str(len(predict_label))




if __name__ == "__main__":
    predict()