'''
Predict word topic and property from LIWC dictionary
Xiyang
'''

import openpyxl

class get_LIWC(object):
    def __init__(self):
        self.dict = {}


    def _read_LIWC(self, file='data/LIWC2007dictionary_poster.xlsx'):
        excel = openpyxl.load_workbook(filename=file, use_iterators=True)
        worksheet = excel.get_active_sheet()
        for col in worksheet.iter_rows():
            print col
        print worksheet





def test_liwc():
    LIWC = get_LIWC()
    LIWC._read_LIWC()


if __name__ == "__main__":
    test_liwc()
