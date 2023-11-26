#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import pickle
from describe import describe
from normalize import min_max_normalize
from softmax import softmax
from statistic import mean
import math
import matplotlib.pyplot as plt
import seaborn as sns
class Logistic_Regression:
    def __init__(self, data, feature_columns, answer_column):
        self.data = data
        self.answer_column = answer_column
        self.feature_columns = feature_columns
        self.answer_list = None 

    def train(self, epochs=2000, lr=0.01):
        train_x = self.data[self.feature_columns]
        m = train_x.shape[0]
        train_y = self.data[self.answer_column]
        train_describe = describe(train_x)
        scaled_x = self.get_scaled_df(train_x, train_describe).to_numpy().T
        scaled_x = np.r_[scaled_x, [np.ones(scaled_x.shape[1])]]
        encoded_y = self.one_hot_encoding(train_y)
        W = np.random.rand(len(self.answer_list), len(self.feature_columns) + 1)
        cost_history = []
        precision_history = []
        for epoch in range(1, epochs + 1):
            z = self.predict(W, scaled_x)
            loss = -1 * encoded_y * np.log(z) + (1 - encoded_y) * np.log(1 - z)
            cost = sum(sum(loss / (m * z.shape[1])))
            cost_history.append(cost)
            dW = np.matmul((z - encoded_y).T, scaled_x.T) / train_x.shape[0]
            W = W - lr * dW
            precision = self.get_precision(z, encoded_y)
            precision_history.append(precision)
            print(f"{epoch} epoch: cost {cost}, precision {precision}")
        with open('parameter.pickle', 'wb') as f:
            pickle.dump(W, f)
            pickle.dump(train_describe, f)
            pickle.dump(self.feature_columns, f)
        sns.lineplot({"cost": cost_history, "precision": precision_history})
        # fig, ax1 = plt.subplots()
        
        # ax1 = sns.lineplot(cost_history, palette="red")
        # ax2 = ax1.twinx()
        # ax2 = sns.lineplot(precision_history, palette="blue")
        plt.show()
        
        
    def get_scaled_df(self, data, data_description):
        scaled_df = pd.DataFrame(columns=self.feature_columns)
        for name in data.columns:
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
        h = np.array(np.zeros((weight.shape[0], x.shape[1])))
        h = np.matmul(weight, x).T
        for i in range(h.shape[0]):
            h[i] = softmax(h[i])
        return h
    
    def get_precision(self, z, y):
        total = 0
        correct = 0
        for index in range(z.shape[0]):
            h = np.argmax(z[index])
            if y[index][h] == 1.0:
                correct += 1
            total += 1
        return correct / total
        
        
        

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
    data_describe = describe(data)
    for column in feature_columns:
        nan_value = mean(data[column])
        data[column] = data[column].fillna(nan_value)
            
            
    answer_column = "Hogwarts House"
    model = Logistic_Regression(data, feature_columns,answer_column)
    model.train()
    