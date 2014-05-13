import os
import matplotlib.pyplot as plt

from csv_dataloader import *
from get_LIWC import *


def draw_LIWC_hist():
    ### Load data
    dataloader = csv_dataloader()
    dataloader.load('output/data_cache.pk')
    print "Read in finished"

    train_id = dataloader.id
    train_data = dataloader.data_retrieve(train_id)
    _, pos_id, neg_id = dataloader.balance(train_id, 'full')
    train_data_pos = dataloader.data_retrieve(pos_id)
    train_data_neg = dataloader.data_retrieve(neg_id)

    tokens = sum(train_data['data'].viewvalues(), [])
    tokens_pos = sum(train_data_pos['data'].viewvalues(), [])
    tokens_neg = sum(train_data_neg['data'].viewvalues(), [])

    ### Calculate LIWC hist
    LIWC = get_LIWC()
    LIWC_hist = LIWC.encode(tokens, normalize=False)
    LIWC_hist_pos = LIWC.encode(tokens_pos, normalize=True)
    LIWC_hist_neg = LIWC.encode(tokens_neg, normalize=True)


    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)
    width = 0.3
    bar0 = ax.bar(np.arange(67)+width, LIWC_hist, width)
    #ax.set_xlabel('Category')
    ax.set_ylabel('Frequency')
    ax.set_title('(a)')
    ax = fig.add_subplot(2,1,2)
    bar1 = ax.bar(np.arange(67)+width, LIWC_hist_pos, width, color='r')
    bar2 = ax.bar(np.arange(67)+2*width, LIWC_hist_neg, width, color='g')

    labels = list(LIWC.category.viewvalues())
    ax.set_label(['Postive', 'Negative'])
    ax.set_xticks(np.arange(67)+2*width)
    ax.set_xticklabels(labels, rotation='vertical')
    ax.set_xlabel('Category')
    ax.set_ylabel('Percentage')
    ax.set_title('(b)')
    ax.grid(True)

    plt.show()

if __name__ == '__main__':
    draw_LIWC_hist()