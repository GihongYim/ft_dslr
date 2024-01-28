import numpy as np


def count(col):
    count = 0
    for element in col:
        if not np.isnan(element):
            count += 1
    return count


def mean(col):
    if len(col) == 0:
        return np.NaN
    num_of_element = count(col)
    if num_of_element == 0:
        return np.NaN
    total = 0.0
    for element in col:
        if not np.isnan(element):
            total += element
    avg = total / float(num_of_element)
    return avg


def std(col):
    num_of_element = count(col)
    if num_of_element == 0:
        return np.NaN
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


def max(col):
    if len(col) == 0:
        return np.NaN
    max_value = col[0]
    for element in col:
        if element > max_value:
            max_value = element
    return max_value


def unique(col):
    dict = {}
    for element in col:
        if element not in dict.keys():
            dict[element] = 0
        dict[element] += 1
    return len(dict)


def top(col):
    dict = {}
    for element in col:
        if element not in dict.keys():
            dict[element] = 0
        dict[element] += 1
    max_element = None
    max_count = 0
    for key in dict.keys():
        if dict[key] > max_count:
            max_element = key
            max_count = dict[key]
    return max_element


def freq(col):
    dict = {}
    for element in col:
        if element not in dict.keys():
            dict[element] = 0
        dict[element] += 1
    max_count = 0
    for key in dict.keys():
        if dict[key] > max_count:
            max_count = dict[key]
    return max_count


def percentile(col, q):
    col = col.to_numpy()
    sorted_col = np.sort(col)

    n = count(sorted_col)
    index = (n - 1) * q / 100

    lower_index = int(np.floor(index))
    fraction = index - lower_index

    if lower_index >= n - 1:
        return sorted_col[n - 1]
    else:
        return sorted_col[lower_index] + fraction \
            * (sorted_col[lower_index + 1] - sorted_col[lower_index])
