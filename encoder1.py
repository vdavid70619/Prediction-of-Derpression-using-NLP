import numpy as np

def encode_feature(samples, ids, encode_method):
    ### Generate Train Positive and Negative Data Encodings Separately
    topics = np.zeros((len(ids), encode_method[0].n_topics))

    if not len(encode_method)<2:
        LIWC_hist = np.zeros((len(ids), 67))
    else:
        LIWC_hist = []

    gender = np.zeros((len(ids), 1))
    time = np.zeros((len(ids), 4))
    i = 0
    for id in ids:
        tokens = samples['ldata'][id]
        topics[i,:] = encode_method[0].encode(tokens)

        if not len(encode_method)<2:
            LIWC_hist[i,:] = encode_method[1].encode(samples['data'][id])

        gender[i] = samples['gender'][id]
        time[i,:] = [samples['time'][id].month, samples['time'][id].day, samples['time'][id].hour, samples['time'][id].minute]
        i +=1
    if len(encode_method)<2:
        encode = np.concatenate((topics, gender, time), axis=1)
    else:
        encode = np.concatenate((topics, gender, time, LIWC_hist), axis=1)
    return encode