import numpy as np

def count(col):
    count = 0
    for element in col:
        if not np.isnan(element):
            count += 1
    return count

def mean(col):
    count = 0
    total = 0.0
    for element in col:
        if not np.isnan(element):
            count += 1
            total += element
    avg = total / float(count)
    return avg

def std(col):
    num_of_element = count(col)
    avg = mean(col)
    total = 0.0
    var = 0.0
    for element in col:
        if not np.isnan(element):
            total += (element - avg) ** 2
    var = total / float(num_of_element - 1)
    standard_deviadtion = var ** (0.5)
    return standard_deviadtion

def min(col):
    if len(col) == 0:
        return np.NaN
    min_value = col[0]
    for element in col:
        if element < min_value:
            min_value = element
    return min_value
        