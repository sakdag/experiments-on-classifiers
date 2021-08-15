# This class uses sklearn-BayesianRidge regression to predict productivity in
# garments_worker_productivity dataset.
import argparse
import os
import time

import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

import src.preprocessing.preprocessing as prep
import src.config.config as conf


def measure_performance(df: pd.DataFrame, number_of_folds: int):

    split_dfs = np.array_split(df, number_of_folds)

    mse_list = list()
    rmse_list = list()
    mape_list = list()

    reg = linear_model.BayesianRidge()

    for i in range(number_of_folds):
        # Get training set by appending elements other than current fold
        train_set = pd.DataFrame()
        for j in range(number_of_folds):
            if j != i:
                train_set = train_set.append(split_dfs[j])

        test_set = split_dfs[i]

        # Default k value for KNeighborsRegressor is 5
        x = train_set.drop(['actual_productivity'], axis=1)
        y = train_set[['actual_productivity']]
        reg.fit(x, y.values.ravel())

        for index, row in test_set.iterrows():
            test_data_x = row.drop(['actual_productivity'])

            test_data_as_list = test_data_x.values.tolist()

            actual = list()
            prediction = reg.predict(np.array(test_data_as_list).reshape(1, -1))
            actual.append(row['actual_productivity'])

            # Calculate MSE
            mse_list.append(mean_squared_error(actual, prediction))

            # Calculate RMSE
            rmse_list.append(mean_squared_error(actual, prediction, squared=False))

            # Calculate MAPE
            mape_list.append(mean_absolute_percentage_error(actual, prediction))

    total_mse = sum(mse_list) / len(mse_list)
    total_rmse = sum(rmse_list) / len(rmse_list)
    total_mape = sum(mape_list) / len(mape_list)

    return total_mse, total_rmse, total_mape


def main():
    dirname = os.path.dirname(__file__)
    dataset_file_path = os.path.join(dirname, conf.PRODUCTIVITY_DATASET_FILE_PATH)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path',
                        default=dataset_file_path,
                        help='absolute path of the dataset you want to use, default: '
                             '{path to project}/data/raw/garments_worker_productivity.csv')
    parser.add_argument('--number_of_folds',
                        type=int,
                        default=3,
                        help='number of folds to use for k-fold cross validation, default: 3')
    parser_args = parser.parse_args()

    df = pd.read_csv(parser_args.dataset_path)
    preprocessed_df = prep.preprocess_productivity_dataset(df)

    print("Running BayesianRidge regression with default settings")

    # Run BayesianRidge regression from sklearn and measure performance
    start_time = time.time()
    bayesian_ridge_mse, bayesian_ridge_rmse, bayesian_ridge_mape = \
        measure_performance(preprocessed_df, parser_args.number_of_folds)
    end_time = time.time()

    print("MSE: ", bayesian_ridge_mse)
    print("RMSE: ", bayesian_ridge_rmse)
    print("MAPE: ", bayesian_ridge_mape)
    print("Time passed: ", end_time - start_time, " seconds\n")


if __name__ == '__main__':
    main()
