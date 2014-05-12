import numpy as np

def encode_feature(samples, ids, encode_method):
    ### Generate Train Positive and Negative Data Encodings Separately
    topics = np.zeros((len(ids), encode_method[1].n_topics))
    gender = np.zeros((len(ids), 1))
    time = np.zeros((len(ids), 4))
    LIWC_hist = np.zeros((len(ids), 67))
    i = 0
    for id in ids:
        tokens = samples['data'][id]
        vec = encode_method[0].batch_convert(tokens)
        topics[i,:] = encode_method[1].encode(vec)
        gender[i] = samples['gender'][id]
        time[i,:] = [samples['time'][id].month, samples['time'][id].day, samples['time'][id].hour, samples['time'][id].minute]
        LIWC_hist[i,:] = encode_method[2].encode(samples['data'][id])
        i +=1
    encode = np.concatenate((topics, gender, time, LIWC_hist), axis=1)
    return encode