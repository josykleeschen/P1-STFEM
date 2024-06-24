import numpy as np


# idea : find the smallest set of elements that have a cumulative residual error bigger than a chosen number.
def doerfler_marking(errorlist, theta):
    idx = errorlist.argsort(axis=0)[::-1]
    list = np.cumsum(errorlist[idx])
    m = np.argmax(list >= list[-1]*theta)
    return idx[:m+1]