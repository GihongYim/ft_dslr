#!/usr/bin/env python3


import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    """
    main function for plotting histogram 
    """
    if len(sys.argv) == 1:
        print(
            "Please enter your filename -> \
                $YOUR_PATH/histogram.py dataset.csv")
        exit(1)
    elif len(sys.argv) >= 3:
        print("invalid input")
        exit(1)

    print(sys.argv[0])
    df = pd.read_csv(sys.argv[1])
    colors = {'Gryffindor': 'red',
              'Slytherin': 'green',
              'Ravenclaw': 'blue',
              'Hufflepuff': 'yellow'}
    # feature_columns = ["Best Hand",
    #                    "Arithmancy",
    #                    "Astronomy",
    #                    "Herbology",
    #                    "Defense Against the Dark Arts",
    #                    "Divination",
    #                    "Muggle Studies",
    #                    "Ancient Runes",
    #                    "History of Magic",
    #                    "Transfiguration",
    #                    "Potions",
    #                    "Care of Magical Creatures",
    #                    "Charms",
    #                    "Flying"
    #                    ]
    plt.figure('2.1 histogram', figsize=(1800/100, 1200/100), dpi=100)
    for i in range(5, len(df.columns)):
        plt.subplot(4,4,i + 1 - 5)
        ax = sns.histplot(data=df,
                      x = df.columns[i],
                      hue = 'Hogwarts House',
                      hue_order=['Slytherin', 'Ravenclaw', 'Gryffindor', 'Hufflepuff'],
                      palette=colors,
                    )
    plt.tight_layout()
    plt.show()
    # index = 0
    # for column in feature_columns:
    #     axs = plt.subplots(ncols=len(feature_columns))
    #     # sns.histplot(data=df,
    #     #              x=column,
    #     #              hue="Hogwarts House",
    #     #              palette=colors,
    #     #              element='poly',
    #     #              ax=axs[index]
    #     #              )
    #     sns.histplot(
    #                  data=df,
    #                  x=column,
    #                  hue="Hogwarts House",
    #                  palette=colors,
    #                  ax=axs[index])
    #     index += 1
    print(df)


if __name__ == "__main__":
    main()
