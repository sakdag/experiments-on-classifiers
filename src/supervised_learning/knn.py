import argparse
import os
import time

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.neighbors import DistanceMetric

import src.preprocessing.preprocessing as prep
import src.config.config as conf

from src.enums.DistanceMetrics import DistanceMetrics


def measure_performance(df: pd.DataFrame, k_value: int, number_of_folds: int, distance_metric):

    # Calculate similarity between train and test data
    # If its one of k neighbors, add it's productivity to
    # current_neighbors map
    split_dfs = np.array_split(df, number_of_folds)

    mse_list = list()
    rmse_list = list()
    mape_list = list()

    if distance_metric.value == DistanceMetrics.MANHATTAN_DISTANCE.value:
        dist = DistanceMetric.get_metric('manhattan')
    else:
        dist = DistanceMetric.get_metric('euclidean')

    for i in range(number_of_folds):
        # Get training set by appending elements other than current fold
        train_set = pd.DataFrame()
        for j in range(number_of_folds):
            if j != i:
                train_set = train_set.append(split_dfs[j])

        train_set_x = train_set.drop(['actual_productivity'], axis=1)
        train_set_y = train_set['actual_productivity']

        test_set = split_dfs[i]
        test_set_x = test_set.drop(['actual_productivity'], axis=1)
        test_set_y = test_set['actual_productivity']

        for test_index, test_row in test_set_x.iterrows():

            actual_productivity = list()
            actual_productivity.append(test_set_y[test_index])
            current_distances = list()

            i = 0
            for train_index, train_row in train_set_x.iterrows():

                first_point = np.array(train_row.values.tolist())
                second_point = np.array(test_row.values.tolist())
                current_distance = dist.pairwise(np.expand_dims(first_point, axis=0),
                                                 np.expand_dims(second_point, axis=0))

                if i < k_value:
                    current_distances.append((current_distance[0][0], train_set_y[train_index]))
                    i += 1
                else:
                    for j in range(k_value):
                        if current_distance[0][0] < current_distances[j][0]:
                            current_distances.insert(j, (current_distance[0][0], train_set_y[train_index]))
                            current_distances.pop()
                            break

            # Calculate weighed average of k nearest neighbors
            sum_of_productivity = 0
            sum_of_weights = 0
            for current_tuple in current_distances:
                sum_of_productivity += float(current_tuple[1]) / float(current_tuple[0])
                sum_of_weights += 1.0 / float(current_tuple[0])

            predicted_productivity = list()
            predicted_productivity.append(sum_of_productivity / sum_of_weights)

            # Calculate MSE
            mse_list.append(mean_squared_error(actual_productivity, predicted_productivity))

            # Calculate RMSE
            rmse_list.append(mean_squared_error(actual_productivity, predicted_productivity, squared=False))

            # Calculate MAPE
            mape_list.append(mean_absolute_percentage_error(actual_productivity, predicted_productivity))

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
    parser.add_argument('--k',
                        type=int,
                        default=3,
                        help='k parameter to use in KNN, default: 3')
    parser_args = parser.parse_args()

    df = pd.read_csv(parser_args.dataset_path)
    preprocessed_df = prep.preprocess_productivity_dataset(df, normalize=True)

    print("Running created knn algorithm with euclidean distance as similarity metric as EUCLIDEAN DISTANCE"
          " and k value: ", parser_args.k)

    # Run knn with created similarity matrix 1 and measure performance
    start_time = time.time()
    knn_1_mse, knn_1_rmse, knn_1_mape = \
        measure_performance(preprocessed_df, parser_args.k, parser_args.number_of_folds,
                            DistanceMetrics.EUCLIDEAN_DISTANCE)
    end_time = time.time()

    print("MSE: ", knn_1_mse)
    print("RMSE: ", knn_1_rmse)
    print("MAPE: ", knn_1_mape)
    print("Time passed: ", end_time - start_time, " seconds\n")

    print("Running created knn algorithm with manhattan distance as similarity metric as MANHATTAN DISTANCE"
          " and k value: ", parser_args.k)

    # Run knn with created similarity matrix 2 and measure performance
    start_time = time.time()
    knn_2_mse, knn_2_rmse, knn_2_mape = \
        measure_performance(preprocessed_df, parser_args.k,
                            parser_args.number_of_folds, DistanceMetrics.MANHATTAN_DISTANCE)
    end_time = time.time()

    print("MSE: ", knn_2_mse)
    print("RMSE: ", knn_2_rmse)
    print("MAPE: ", knn_2_mape)
    print("Time passed: ", end_time - start_time, " seconds\n")


if __name__ == '__main__':
    main()
