import pandas
import pandas as pd
import numpy as np

from src import knn

if __name__ == '__main__':

    k_value = 5
    number_of_folds = 3

    df = pd.read_csv("C:/Users/serha/Desktop/Ms_1.2/CEng514-DM/HW2/experiments-on-classifiers/"
                     "resources/garments_worker_productivity.csv")
    print(df)

    shuffled_df = df.sample(frac=1)
    print(shuffled_df)

    split_dfs = np.array_split(shuffled_df, number_of_folds)

    columns_to_use = {"smv", "team"}

    for i in range(number_of_folds):
        # Get training set by appending elements other than current fold
        rest = pandas.DataFrame()
        for j in range(number_of_folds):
            if j != i:
                rest = rest.append(split_dfs[j])

        print(split_dfs[i])
        print(rest)

        # Run knn and measure performance
        knn.measure_performance(rest, split_dfs[i], k_value, columns_to_use)


