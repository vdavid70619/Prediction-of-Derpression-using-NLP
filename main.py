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

    data={}
    dataloader = csv_dataloader()
    data, label, score = dataloader.read_csv()
    dataloader.summary()
    nfolds = dataloader.nfold(5)

    words = str(data.viewvalues())
    tokens = nltk.WordPunctTokenizer().tokenize(words.lower())
    tokens = list(set(tokens))

    tokens = preprocess(tokens)

    tokensfile = open('output/tokens.txt', 'wr')
    tokensfile.write(str(tokens))

    print len(tokens)

    text8file = open('data/text8.txt', 'r')
    text8 = text8file.read();
    print len(text8)



if __name__ == "__main__":
    main()
