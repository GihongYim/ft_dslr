#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def main():
    if len(sys.argv) != 2:
        print("input parameter")
        return
    try:
        df = pd.read_csv(sys.argv[1])
    except Exception as e:
        print(e)
        return 
    hogwart_colors = {'Gryffindor': '#ae0001', 'Slytherin':'#2a623d', 'Ravenclaw': '#222f5b', 'Hufflepuff':'#f0c75e'}
    feature_columns = ["Hogwarts House",
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
    feature_df = df[feature_columns]
    sns.pairplot(data=feature_df, hue="Hogwarts House", palette=hogwart_colors)
    plt.show()
if __name__ == "__main__":
    main()