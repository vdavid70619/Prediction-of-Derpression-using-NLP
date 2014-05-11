'''
    Data loader for final project
    Xiyang
'''

import csv
import math

import random
import cPickle as pickle
import dateutil.parser as parser

class csv_dataloader(object):
    #### private
    _datafile_path = 'data/train_statuses.csv'
    _labelfile_path = 'data/trainScores.csv'
    _extrafile_path = ''
    _limit = -1
    _nolabel = False
    _extra = False

    #### public
    id = []
    data = {}
    ldata = {}
    label = {}
    score = {}
    gender = {}
    time = {}

    def __init__(self, datafile='data/train_statuses.csv', labelfile='data/fixed_train_gender_class.csv',
                extrafile=None, limit=-1, nolabel=False, extra=False):

        self._datafile_path = datafile
        self._labelfile_path = labelfile
        self._extrafile_path = extrafile
        self._nolabel = nolabel
        self._extra = extra
        self._limit = limit


    def read_csv(self, applyfun=(lambda x: x+' '), verbose=False):
        '''
        Read in csv and do preprocessing on it
        applyfun: string -> anything
        '''

        ### Read train data from csv file
        with open(self._datafile_path, 'rb') as data_file:
            data_reader = csv.reader(data_file, dialect="excel")
            counts = 0
            for row in data_reader:
                if verbose and divmod(counts, 10000)==0:
                    print 'line read '+ str(counts)

                if counts==0:
                    counts += 1
                    continue

                counts += 1
                #print  "ids: " + row[0] + " posts: " + row[1] + " data: " + row[2]

                if row[0] not in self.data:
                    self.data[row[0]] = applyfun(row[1])
                    self.ldata[row[0]] = [applyfun(row[1])]
                    self.id.append(row[0])
                else:
                    self.data[row[0]] += applyfun(row[1]) ## append space to sperate lines
                    self.ldata[row[0]].append(applyfun(row[1]))

                if counts==self._limit:
                    break

        ### Read train label and score if there is any
        if self._nolabel==False:
            with open(self._labelfile_path, 'rb') as label_file:
                label_reader = csv.reader(label_file, dialect="excel")
                counts = 0
                for row in label_reader:
                    if counts==0:
                        counts += 1
                        continue

                    counts += 1
                    #print  "ids: " + row[0] + " date: " + row[1] + " score: " + row[2] + " label: " + row[3]

                    self.label[row[0]] = int(row[-1]=="+")
                    self.score[row[0]] = int(row[-2])

                    if counts==self._limit:
                        break

        ### Read train label and score if there is any
        if self._extra==True:
            with open(self._extrafile_path, 'rb') as extra_file:
                extra_reader = csv.reader(extra_file, dialect="excel")
                counts = 0
                for row in extra_reader:
                    if counts==0:
                        counts += 1
                        continue

                    counts += 1
                    #print  "ids: " + row[0] + " date: " + row[1] + " score: " + row[2] + " label: " + row[3]

                    self.gender[row[0]] = int(row[2]=="1")
                    self.time[row[0]] = parser.parse(row[1])

                    if counts==self._limit:
                        break

        return self.data, self.label, self.score

    def nfold(self, n):

        folds = {}
        for i in range(n):
            folds[i] = []

        pos = []
        neg = []

        for key, label in self.label.items():
            if label==1:
                pos.append(key)
            else:
                neg.append(key)

        pos = random.sample(pos, len(pos))
        neg = random.sample(neg, len(neg))

        pos_fold_count = int(math.ceil(len(pos)*1.0/n))
        neg_fold_count = int(math.ceil(len(neg)*1.0/n))

        for i, key in enumerate(pos):
            folds[(i/pos_fold_count)].append(key)

        for i, key in enumerate(neg):
            folds[(i/neg_fold_count)].append(key)

        return folds

    def balance(self, ids, method='Downsample', K=2):
        method = method.lower()

        neg = []
        pos = []

        for id in ids:
            if self.label[id] == 1:
                pos.append(id)
            else:
                neg.append(id)

        ratio = len(neg)/len(pos)
        neg = random.sample(neg, len(neg))

        if method=='downsample':
            neg = neg[:min(K, ratio)*len(pos)]

        return pos+neg, pos, neg

    def batch_retrieve(self, ids):
        batch_label={}
        batch_score={}
        batch_data={}
        batch_ldata={}
        batch_gender={}
        batch_time={}

        for id in ids:
            batch_data[id] = self.data[id]
            batch_ldata[id] = self.ldata[id]
            if not self._nolabel:
                batch_label[id] = self.label[id]
                batch_score[id] = self.score[id]
            if self._extra:
                batch_gender[id] = self.gender[id]
                batch_time[id] = self.time[id]

        return batch_data, batch_ldata, batch_label, batch_score, batch_gender, batch_time

    def summary(self):
        print "Total Data size: " + str(len(self.data.keys()))
        print "Positive Data size: " + str(sum(self.label.values()))


    def load(self, filename):
        with open(filename, 'rb') as input:
            cache = pickle.load(input)
            self.data = cache['data']
            self.ldata = cache['ldata']
            self.label = cache['label']
            self.score = cache['score']
            self.gender = cache['gender']
            self.time = cache['time']
            self.id = cache['id']

    def save(self, filename):
        cache = {}
        cache['data'] = self.data
        cache['ldata'] = self.ldata
        cache['label'] = self.label
        cache['score'] = self.score
        cache['gender'] = self.gender
        cache['time'] = self.time
        cache['id'] = self.id

        with open(filename, 'wb+') as output:
            ## save a class object to a file using pickle
            pickle.dump(cache, output, pickle.HIGHEST_PROTOCOL)