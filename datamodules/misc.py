import numpy as np
import pandas as pd

def trip_advisor_label_map(rating):
    return int(rating)-1

def beer_advocate_label_map(rating):
    return int(rating)-1

def social_news_label_map(label):
    return 1 if label == 2 else 0
        
def truncate(x, max_len):
    return x[:max_len]

def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    """Pad or/and truncate the input sequence"""
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x