import numpy as np
import pandas as pd
from normalize import min_max_normalize
from softmax import softmax
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from describe import describe
from statistic import mean


class Logistic_Regression:
    def __init__(self, data=None, feature_columns=None, answer_column=None):
        self.data = data
        self.answer_column = answer_column
        self.feature_columns = feature_columns
        self.answer_list = None 
        self.W = None
        self.train_describe = None

    def train(self, epochs=2000, lr=0.01, batch_size=300):
        # batch_size = 0 -> batch_size = m
        train_x = self.data[self.feature_columns]
        m = train_x.shape[0]
        train_y = self.data[self.answer_column]
        self.train_describe = describe(train_x)
        scaled_x = self.get_scaled_df(train_x, self.train_describe).to_numpy()
        scaled_x = np.c_[scaled_x, np.ones(scaled_x.shape[0])]
        encoded_y, self.answer_list = self.one_hot_encoding(train_y)
        self.W = np.random.rand(len(self.feature_columns) + 1,
                                len(self.answer_list))
        cost_history = []
        precision_history = []
        if batch_size == 0:
            batch_size = m
        num_of_batch = m // batch_size
        for epoch in range(1, epochs + 1):
            # iteration for batch
            for i in range(num_of_batch):
                batch_x = scaled_x[i * batch_size: (i + 1) * batch_size]
                batch_y = encoded_y[i * batch_size: (i + 1) * batch_size]
                z = self.predict(self.W, batch_x)
                loss = -1 * batch_y * np.log(z) + (1 - batch_y) * np.log(1 - z)
                dW = np.matmul((z - batch_y).T, batch_x) / batch_x.shape[0]
                self.W = self.W - lr * dW.T
            # batch remainder
            batch_x = scaled_x[(num_of_batch - 1) * batch_size:]
            batch_y = encoded_y[(num_of_batch - 1) * batch_size:]
            z = self.predict(self.W, batch_x)
            dW = np.matmul((z - batch_y).T, batch_x) / batch_x.shape[0]
            self.W = self.W - lr * dW.T

            total_z = self.predict(self.W, scaled_x)
            loss = -1 * encoded_y * np.log(total_z) + \
                (1 - encoded_y) * np.log(1 - total_z)
            cost = sum(sum(loss / (scaled_x.shape[0] * total_z.shape[1])))
            cost_history.append(cost)
            precision = self.get_precision(total_z, encoded_y)
            precision_history.append(precision)
            print(f"{epoch} epoch: cost {cost}, precision {precision}")
        with open('parameter.pickle', 'wb') as f:
            pickle.dump(self.W, f)
            pickle.dump(self.train_describe, f)
            pickle.dump(self.feature_columns, f)
            pickle.dump(self.answer_list, f)
        sns.lineplot({"cost": cost_history, "precision": precision_history})
        plt.show()

    def get_scaled_df(self, data, data_description):
        scaled_df = pd.DataFrame(columns=self.feature_columns)
        for name in data.columns:
            column_min = data_description[name]['min']
            column_max = data_description[name]['max']
            scaled_df[name] = min_max_normalize(
                data[name], column_min, column_max)
        return scaled_df

    def one_hot_encoding(self, train_y):
        answer_list = np.unique(train_y)
        encoded_column = np.zeros((len(train_y), len(answer_list)))
        for row in range(len(train_y)):
            for col in range(len(answer_list)):
                if train_y[row] == answer_list[col]:
                    encoded_column[row][col] = 1
        return encoded_column, answer_list

    def predict(self, weight, x):
        h = np.array(np.zeros((weight.shape[0], x.shape[1])))
        h = np.matmul(x, weight)
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

    def get_parameter(self, filename):
        with open(filename, 'rb') as f:
            self.W = pickle.load(f)
            self.train_describe = pickle.load(f)
            self.feature_columns = pickle.load(f)
            self.answer_list = pickle.load(f)

    def predict_answer(self, testfile, outputfile):
        self.data = pd.read_csv(testfile)
        for column in self.feature_columns:
            mean_value = mean(self.data[column])
            self.data[column] = self.data[column].fillna(mean_value)
        train_x = self.data[self.feature_columns]
        m = train_x.shape[0]
        scaled_x = self.get_scaled_df(train_x, self.train_describe).to_numpy()
        scaled_x = np.c_[scaled_x, np.ones(scaled_x.shape[0])]
        h = self.predict(self.W, scaled_x)
        answer = np.empty(m, dtype='object')
        for index in range(m):
            answer[index] = self.answer_list[np.argmax(h[index])]
        answer_df = pd.DataFrame(
            {'Index': range(1, m + 1), 'Hogwarts House': answer})
        print(answer_df)
        answer_df.to_csv(outputfile, index=False)
