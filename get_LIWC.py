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


    def _read_LIWC(self, file='data/LIWC2007dictionary_poster.xlsx'):
        '''
        Do the dirty & hard-coding way to extract information from the poster
        '''

        Catogries_indx = [3,]
        excel = openpyxl.load_workbook(filename=file, use_iterators=True)
        worksheet = excel.get_active_sheet()
        row_i = 1
        category_i = 0
        category_lengths = {}
        temp_length=0
        pre_style_id = None # hard coding for the specific poster, change may be required for others
        for row in worksheet.iter_rows():
            print row_i
            if row_i == 1:
                pass
            elif row_i == 2:
                print row
                for cell in row:
                    if  cell.style_id != pre_style_id:
                        category_i += 1
                        category_lengths[category_i] = 1
                    else:
                        category_lengths[category_i] +=1

                    pre_style_id = cell.style_id
                    print cell.style_id

                print 'done'
                print category_lengths
                break
            else:
                pass
            row_i += 1





def test_liwc():
    LIWC = get_LIWC()
    LIWC._read_LIWC()


if __name__ == "__main__":
    test_liwc()
