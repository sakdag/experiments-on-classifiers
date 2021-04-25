import numpy
import numpy as np
import pandas
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.neighbors import DistanceMetric
from src.main import DistanceMetrics


def measure_performance(df: pandas.DataFrame, k_value: int, number_of_folds: int, distance_metric):

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
        train_set = pandas.DataFrame()
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

                first_point = numpy.array(train_row.values.tolist())
                second_point = numpy.array(test_row.values.tolist())
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
