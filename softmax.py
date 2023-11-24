import numpy as np

def softmax(z):
    print(np.sum(np.exp(z), axis=1))
    return np.exp(z) / np.sum(np.exp(z), axis=1)