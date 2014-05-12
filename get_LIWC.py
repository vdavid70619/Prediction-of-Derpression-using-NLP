'''
Extract word topic and property from LIWC dictionary
Xiyang

This package is provided AS-IS

Copyright by LIWC and Professor Philip Resnik
'''

import openpyxl
import cPickle as pickle
import numpy as np
from pprint import pprint

class get_LIWC(object):
    def __init__(self, verbose=False):
        self.dict = {}
        self.category = {}
        self.dict = {}
        self.verbose = verbose
        self._read_LIWC()


    def _read_LIWC(self, file='data/LIWC2007dictionary_poster.xlsx'):
        '''
        Do the dirty & hard-coding way to extract information from the poster
        '''

        excel = openpyxl.load_workbook(filename=file, use_iterators=True)
        worksheet = excel.get_active_sheet()
        category_lengths = {}
        # Do word extraction based on color. :(
        row_i = 1
        for row in worksheet.iter_rows():
            if row_i == 1:
                pass
            elif row_i == 2:
                category_i = 0
                pre_style_id = None # hard coding for the specific poster, change may be required for others
                for cell in row:
                    if cell.style_id != pre_style_id:
                        category_i += 1
                        category_lengths[category_i] = 1
                    else:
                        category_lengths[category_i] +=1
                    pre_style_id = cell.style_id
                    if self.verbose:
                        print cell

                if self.verbose:
                    print category_lengths
            elif row_i == 3:
                col_j = 0
                category_i = 1
                for cell in row:
                    if col_j >= category_lengths[category_i]:
                        category_i += 1
                        col_j = 0

                    if cell.internal_value is not None:
                        self.category[category_i] = cell.internal_value

                    col_j += 1
            else:
                col_j = 0
                category_i = 1
                for cell in row:
                    if col_j >= category_lengths[category_i]:
                        category_i += 1
                        col_j = 0

                    if cell.internal_value is not None:
                        word = cell.internal_value.replace('*','')
                        if self.dict.has_key(word):
                            self.dict[word].append(category_i)
                        else:
                            self.dict[word] = [category_i]

                    col_j += 1

            row_i += 1
        if self.verbose:
            print '#Total categories: ' + str(len(self.dict.viewkeys()))
            print 'All categories: ' + str(self.category)
            pprint(self.dict)

    def __getitem__(self, word):
        word = word.lower()
        if self.dict.has_key(word):
            return self.dict[word]
        else:
            return []

    def encode(self, words, normalize=True):
        hist = np.zeros(67) ## <TODO> Bug here
        for word in words:
            categories = self[word]
            category_size = len(categories)
            for category in categories:
                hist[category-1] += 1.0/category_size
        if normalize:
            hist = hist/np.sum(hist + 1e-6, axis=0)
        return hist

    def load(self, filename):
        with open(filename, 'rb') as input:
            self.dixt = pickle.load(input)

    def save(self, filename):
        with open(filename, 'wb+') as output:
            ## save a class object to a file using pickle
            pickle.dump(self.dict, output, pickle.HIGHEST_PROTOCOL)

def test_liwc():
    LIWC = get_LIWC(verbose=True)
    LIWC.save('LIWC_cache.pk')
    print LIWC.calculate_hist(['I', 'am', 'correct', 'ohwell'], normalize=True)
    print LIWC['I']
    print LIWC['am']
    print LIWC['correct']
    print LIWC['ohwell']

# if __name__ == "__main__":
#     test_liwc()
