import os
import matplotlib

from csv_dataloader import *
from get_LIWC import *


def draw_LIWC_hist():
    ### Load data
    dataloader = csv_dataloader(extrafile='data/fixed_train_gender_class.csv', extra=True)
    if not os.path.exists('output/data_cache.pk'):
        dataloader.read_csv(applyfun=preprocess)
        dataloader.save('output/data_cache.pk')
    else:
        dataloader.load('output/data_cache.pk')
    dataloader.summary()
    print "Read in finished"

    train_id = dataloader.id
    train_data = dataloader.data_retrieve(train_id)
    _, pos_id, neg_id = dataloader.balance(train_id, 'full')

    ### Calculate LIWC hist
    LIWC = get_LIWC()
    #print LIWC.calculate_hist(tokens, normalize=False)




if __name__ == '__main__':
    draw_LIWC_hist()