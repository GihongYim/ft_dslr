#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import pickle
from describe import describe
from normalize import min_max_normalize
from softmax import softmax

class Logistic_Regression:
    def __init__(self, data, feature_columns, answer_column):
        self.data = data
        self.answer_column = answer_column
        self.feature_columns = feature_columns
        self.answer_list = None 

    def train(self, epoch=1000, lr=0.003):
        train_x = self.data[self.feature_columns]
        train_y = self.data[self.answer_column]
        train_describe = describe(train_x)
        scaled_x = self.get_scaled_df(train_x, train_describe).to_numpy().T
        encoded_y = self.one_hot_encoding(train_y)
        weights = np.random.rand(len(self.answer_list), len(self.feature_columns) + 1)
        print(self.predict(weights, scaled_x))
        
        
    def get_scaled_df(self, data, data_description):
        scaled_df = pd.DataFrame(columns=self.feature_columns)
        for name in self.feature_columns:
            column_min = data_description[name]['min']
            column_max = data_description[name]['max']
            scaled_df[name] = min_max_normalize(data[name], column_min, column_max)
        return scaled_df
    
    def one_hot_encoding(self, train_y):
        self.answer_list = np.unique(train_y)
        encoded_column = np.zeros((len(train_y), len(self.answer_list)))
        for row in range(len(train_y)):
            for col in range(len(self.answer_list)):
                if train_y[row] == self.answer_list[col]:
                    encoded_column[row][col] = 1
        return encoded_column
    
    def predict(self, weight, x):
        h = np.array((weight.shape[1]))
        x_b = np.r_[x, [np.ones(1600)]]
        h = np.matmul(weight, x_b)
        for i in range(len(h)):
            h[i] = np.matmul(weight[i], x_b)
        print(softmax(h.T).shape)
        
        
        
        

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
    answer_column = "Hogwarts House"
    model = Logistic_Regression(data, feature_columns,answer_column)
    model.train()
    