#!/usr/bin/env python3

import sys
import pandas as pd
from describe import describe
from statistic import mean
from Logistic_Regression import Logistic_Regression

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("csv filename not found")
        exit(0)
    data = pd.read_csv(sys.argv[1])
    feature_columns = [
            "Arithmancy",
            "Astronomy",
            "Herbology",
            # "Defense Against the Dark Arts",
            "Divination",
            "Muggle Studies",
            "Ancient Runes",
            "History of Magic",
            # "Transfiguration",
            "Potions",
            # "Care of Magical Creatures",
            "Charms",
            "Flying"
        ]
    data_describe = describe(data)
    for column in feature_columns:
        mean_value = mean(data[column])
        data[column] = data[column].fillna(mean_value)

    answer_column = "Hogwarts House"
    model = Logistic_Regression(data, feature_columns, answer_column)
    model.train(batch_size=100)
