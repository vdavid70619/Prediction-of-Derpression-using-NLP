"""
    Main function
"""

import re
import nltk
import gensim

from csv_dataloader import *


def has_pattern(word, pattern):
    pass


def preprocess(tokens):

    stopwords = nltk.corpus.stopwords.words('english')
    tokens = [w for w in tokens if w not in stopwords]
    tokens = [w for w in tokens if len(w)<20]
    tokens = [w for w in tokens if re.match('\W+',w)==None]

    return tokens

def main():

    NFOLD = 5

    dataloader = csv_dataloader()
    data, label, score = dataloader.read_csv()
    dataloader.summary()
    nfolds = dataloader.nfold(NFOLD)

    words = str(data.viewvalues())
    tokens = nltk.WordPunctTokenizer().tokenize(words.lower())
    tokens = list(set(tokens))

    tokens = preprocess(tokens)

    # tokensfile = open('output/tokens.txt', 'wr')
    # tokensfile.write(str(tokens))

    train_id = []
    for i in range(NFOLD-2):
        train_id += nfolds[i]

    test_id = nfolds[NFOLD-1]

    print len(train_id)
    print train_id
    print len(test_id)
    print test_id





if __name__ == "__main__":
    main()
