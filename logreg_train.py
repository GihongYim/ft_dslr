#!/usr/bin/env python3


import sys
import numpy as np
import pandas as pd
import pickle

class Logistic_Regression:
    def __init__(self, data):
        self.data = data
        
    def train(self, epoch=0.0, lr=0.0):
        
        pass
    
    



    
    


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("csv filename not found")
        exit(0)
    data = pd.read_csv(sys.argv[1])
    model = Logistic_Regression(data)
    