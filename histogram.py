#!/usr/bin/env python3


import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def main():
    if len(sys.argv) == 1:
        print("Please enter your filename -> $YOUR_PATH/histogram.py dataset.csv")
        exit(1)
    elif len(sys.argv) >= 3:
        print("invalid input")
        exit(1)
    
    print(sys.argv[0])
    df = pd.read_csv(sys.argv[1])
    hogwart_name = df['Hogwarts House'].unique()
    hogwart_name_df = {}
    colors = {'Gryffindor':'red', 'Slytherin':'green', 'Ravenclaw':'blue', 'Hufflepuff':'yellow'}
    for name in hogwart_name:
        hogwart_name_df[name] = df[df['Hogwarts House'] == name]
    num_of_row = len(hogwart_name)
    num_of_column = hogwart_name_df[hogwart_name[0]].shape[1]
    print(num_of_row, num_of_column)
    index = 1
    hogwart_index = 0
    for name in hogwart_name_df:
        for column in hogwart_name_df[name]:
            plt.subplot(num_of_row, num_of_column, index)
            sns.histplot(hogwart_name_df[name][column], color=colors[name]).set_title(name + ' ' + column)
            
            index += 1
        hogwart_index += 1
    plt.show()
    
    
    
    
if __name__ == "__main__":
    main()