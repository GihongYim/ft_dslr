#!/usr/bin/env python3


import sys
import numpy as np
import pandas as pd
import pickle
from describe import describe
from normalize import min_max_normalize

class Logistic_Regression:
    def __init__(self, data, feature_columns):
        self.data = data
        self.feature_columns = feature_columns
        

    def train(self, epoch=1000, lr=0.003):
        train_df = self.data[self.feature_columns]
        train_describe = describe(train_df)
        scaled_df = self.get_scaled_df(train_df, train_describe)
        print(scaled_df)
        weight = np.zeros(len(feature_columns))
        

    def get_scaled_df(self, data, data_description):
        scaled_df = pd.DataFrame(columns=self.feature_columns)
        for name in self.feature_columns:
            column_min = data_description[name]['min']
            column_max = data_description[name]['max']
            scaled_df[name] = min_max_normalize(data[name], column_min, column_max)
        return scaled_df
    
    

    



    
    


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("csv filename not found")
        exit(0)
    data = pd.read_csv(sys.argv[1])
    feature_columns = [
            "Arithmancy", 
            "Astronomy", 
            "Herbology",
            "Defense Against the Dark Arts", 
            "Divination", 
            "Muggle Studies", 
            "Ancient Runes", 
            "History of Magic", 
            "Transfiguration", 
            "Potions", 
            "Care of Magical Creatures",
            "Charms", 
            "Flying" 
        ]
    model = Logistic_Regression(data, feature_columns)
    model.train()
    