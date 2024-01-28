#!/usr/bin/env python3

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# feature list
# Index, (x) :
# Hogwarts House, (x) : answer
# First Name, (x)
# Last Name, (x)
# Birthday, (x)
# Best Hand, (x) almost same
# Arithmancy, (o)
# Astronomy, (o)
# Herbology,(o)
# Defense Against the Dark Arts, (o)
# Divination, (o)
# Muggle Studies, (o)
# Ancient Runes, (o)
# History of Magic, (o)
# Transfiguration, (o)
# Potions, (o)
# Care of Magical Creatures,Charms, (o)
# Flying (o)


def main():
    if len(sys.argv) != 2:
        print("input parameter")
        return
    try:
        df = pd.read_csv(sys.argv[1])
    except Exception as e:
        print(e)
        return
    hogwart_colors = {'Gryffindor': '#ae0001',
                      'Slytherin': '#2a623d',
                      'Ravenclaw': '#222f5b',
                      'Hufflepuff': '#f0c75e'}
    feature_columns = ["Best Hand",
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
    fig, axs = plt.subplots(len(feature_columns), len(feature_columns))
    row_index, column_index = 0, 0
    for row_feature in feature_columns:
        column_index = 0
        for column_feature in feature_columns:
            if row_feature == column_feature:
                column_index += 1
                continue
            sns.scatterplot(data=df,
                            x=column_feature,
                            y=row_feature,
                            hue='Hogwarts House',
                            palette=hogwart_colors,
                            ax=axs[row_index][column_index],
                            legend=False)
            column_index += 1
        row_index += 1
    plt.show()


if __name__ == "__main__":
    main()
