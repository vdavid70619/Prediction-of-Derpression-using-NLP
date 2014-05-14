'''
Preprocessing Step
'''

import re
import nltk

def preprocess(words):

    words = unicode(words, errors='ignore') #This is for gensim
    tokens = nltk.WordPunctTokenizer().tokenize(words.lower())
    # tokens = list(set(tokens))
    self_defined = ['propfemale', 'propmale', 'propfirst']
    stopwords = nltk.corpus.stopwords.words('english') + self_defined
    tokens = [w for w in tokens if w not in stopwords]
    tokens = [w for w in tokens if len(w)<20 and len(w)>2]
    tokens = [w for w in tokens if re.match('\W+',w)==None]
    return tokens