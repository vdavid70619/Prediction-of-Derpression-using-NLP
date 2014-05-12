'''
    Convert word to numpy standard sparse vector representation for final project
    Xiyang
'''

import gensim
import numpy as np
import os

class get_sentences(object):
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        if os.path.isfile(self.fname):
            for line in open(self.fname):
                yield line.split()

        elif os.path.isdir(self.fname):
            for fname in os.listdir(self.dirname):
                for line in open(os.path.join(self.dirname, fname)):
                    yield line.split()


class get_word2vec(object):
    def __init__(self, model=None, binary=None, size=100):
        if model:
            self.vdim = size
            if binary:
                self.model = gensim.models.Word2Vec.load_word2vec_format(model, binary=True)
            else:
                self.model = gensim.models.Word2Vec.load(model)
        else:
            self.model = gensim.models.Word2Vec()

    def train(self, file, vec_size=100, workers=4):
        sentences = get_sentences(file)
        self.model = gensim.models.Word2Vec(sentences=sentences, size=vec_size, workers=workers)
        self.vdim = vec_size

    def save(self,file):
        self.model.save(file)

    def __getitem__(self, word):
        if self.model.__contains__(word):
            return self.model[word]
        else:
            return None

    def batch_convert(self, words, verbose=True):

        batch_size = 100000.0
        total_size = 0.0
        i=0
        batch_vectors = np.zeros((0, self.vdim))
        for word in words:
            vec = self[word]
            if vec!=None:
                if i%batch_size==0:
                    total_size += batch_size
                    if verbose:
                        print str(total_size) + '.',
                    batch_vectors = np.resize(batch_vectors, (total_size, self.vdim))
                    batch_vectors[i,:] = vec
                else:
                    batch_vectors[i,:] = vec

                i += 1

        return batch_vectors[0:i]


def get_word2vec_test():

    word2vec = get_word2vec(model='data/GoogleNews-vectors-negative300.bin', binary=True, size=300)
    print word2vec['computer']
    print word2vec['smart']
    print word2vec.model.most_similar(positive=['computer'], topn=25)
    print word2vec.batch_convert(['computer','compputer','smart'])

#
# if __name__ == "__main__":
#     get_word2vec_test()