import pandas as pd
import numpy as np
import statistic


def describe(df):
    """
    describe main statistic
    
    mean, std, max, min

    Args:
        df (pd.DataFrame): pandas DataFrame datatype
    """
    numeric_list = []
    for col in df:
        if df[col].dtype == 'float64' or df[col].dtype == 'int64':
            numeric_list.append(col)
    statistic_df = pd.DataFrame(columns=numeric_list)
    count = {}
    for col in numeric_list:
        count[col] = statistic.count(df[col])
    statistic_df.loc['count'] = count
    mean = {}
    for col in numeric_list:
        mean[col] = statistic.mean(df[col])
    statistic_df.loc['mean'] = mean
    std = {}
    for col in numeric_list:
        std[col] = statistic.std(df[col])
    statistic_df.loc['std'] = std
    min = {}
    for col in numeric_list:
        min[col] = statistic.min(df[col])
    statistic_df.loc['min'] = min
    q25 = {}
    for col in numeric_list:
        q25[col] = np.nanpercentile(df[col], q=25)
    statistic_df.loc['25%'] = q25
    q50 = {}
    for col in numeric_list:
        q50[col] = np.nanpercentile(df[col], q=50)
    statistic_df.loc['50%'] = q50
    q75 = {}
    for col in numeric_list:
        q75[col] = np.nanpercentile(df[col], q=75)
    statistic_df.loc['75%'] = q75
    max = {}
    for col in numeric_list:
        max[col] = np.max(df[col])
    statistic_df.loc['max'] = max
    # print(mean)
    return statistic_df
    