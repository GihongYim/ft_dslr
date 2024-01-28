import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":
    df = pd.read_csv("dataset_train.csv", index_col='Index')
    print(df)
    sns.pairplot(df, hue="Hogwarts House")
    plt.show()
