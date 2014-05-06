'''
Extract word topic and property from LIWC dictionary
Xiyang

This package is provided AS-IS

Copyright by LIWC and Professor Philip Resnik
'''

import openpyxl

class get_LIWC(object):
    def __init__(self):
        self.dict = {}
        self.category = {}
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

                print sum(category_lengths.viewvalues())
            elif row_i == 3:
                category_i = 0
                pre_style_id = None # hard coding for the specific poster, change may be required for others
                for cell in row:
                    if cell.style_id != pre_style_id:
                        category_i += 1

                    if cell.internal_value is not None:
                        print cell.internal_value
                        self.category[category_i] = cell.internal_value

                    pre_style_id = cell.style_id

                #print category
            else:
                category_i = 0
                pre_style_id = None # hard coding for the specific poster, change may be required for others
                for cell in row:
                    if cell.style_id != pre_style_id:
                        category_i += 1

                    if cell.internal_value is not None:
                        print cell.internal_value
                        if self.dict.has_key(category_i):
                            self.dict[category_i].append(cell.internal_value)
                        else:
                            self.dict[category_i] = [cell.internal_value]


                    pre_style_id = cell.style_id
                #print dictionary
            row_i += 1

        print '#Total categories: ' + str(len(self.dict.viewkeys()))
        print self.dict.viewvalues()

    def __getitem__(self, word):
        category = [k for k, v in self.dict.iteritems() if word in v];
        if category is None:
            return -1
        else:
            return category


def test_liwc():
    LIWC = get_LIWC()
    print LIWC['dine']


if __name__ == "__main__":
    test_liwc()
