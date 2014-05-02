'''
    Data loader for final project
    Xiyang
'''

import csv
import math

import random

class csv_dataloader:
    #### private
    _datafile_path = 'data/train_statuses.csv'
    _labelfile_path = 'data/trainScores.csv'
    _limit = -1
    _nolabel = False

    #### public
    data = {}
    ldata = {}
    label = {}
    score = {}

    def __init__(self, datafile='data/train_statuses.csv', labelfile='data/fixed_train_gender_class.csv', limit=-1, nolabel=False):
        self._datafile_path = datafile
        self._labelfile_path = labelfile
        self._nolabel = nolabel
        self._limit = limit
    
    def read_csv(self):
        ### Read train data from csv file
        with open(self._datafile_path, 'rb') as data_file:
            data_reader = csv.reader(data_file, dialect="excel")
            counts = 0
            for row in data_reader:
                if counts==0:
                    counts += 1
                    continue

                counts += 1
                #print  "ids: " + row[0] + " posts: " + row[1] + " data: " + row[2]

                if row[0] not in self.data:
                    self.data[row[0]] = row[1]
                    self.ldata[row[0]] = [row[1]]
                else:
                    self.data[row[0]] += (" " + row[1]) ## append space to sperate lines
                    self.ldata[row[0]].append(row[1])

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

        return pos+neg

    def batch_retrieve(self, ids):
        batch_label={}
        batch_score={}
        batch_data={}

        for id in ids:
            batch_data[id] = self.data[id]
            batch_label[id] = self.label[id]
            batch_score[id] = self.score[id]

        return batch_data, batch_label, batch_score


    def summary(self):
        print "Total Data size: " + str(len(self.data.keys()))
        print "Positive Data size: " + str(sum(self.label.values()))


