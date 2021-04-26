import time
from enum import Enum
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from src import knn, k_neighbors_regressor, decision_tree_regressor, naive_bayes


class DistanceMetrics(Enum):
    MANHATTAN_DISTANCE = 1
    EUCLIDEAN_DISTANCE = 2


if __name__ == '__main__':
    number_of_folds = 3

    df = pd.read_csv("C:/Users/sakdag/Desktop/Ms_1.2/CEng514/HW2/experiments-on-classifiers/resources/"
                     "garments_worker_productivity.csv")
    print(df)

    # Eliminate not numeric and missing element columns for now
    df_selected = df.drop(['date', 'targeted_productivity', 'idle_time',
                           'idle_men', 'no_of_style_change'], axis=1)
    print(df_selected)

    # Fill empty values in column wip with mean of wip values
    df_selected['wip'].fillna((df_selected['wip'].mean()), inplace=True)
    print(df_selected)

    # Shuffle elements of dataframe
    shuffled_df = df_selected.sample(frac=1).reset_index(drop=True)
    print(shuffled_df)

    shuffled_df = df_selected.copy()

    # Convert string valued columns to numeric values
    # There are whitespaces in department column that interferes with encoding, so strip them first
    shuffled_df['department'] = shuffled_df['department'].str.strip()
    enc = OrdinalEncoder()
    enc.fit(shuffled_df[['department', 'quarter', 'day']])
    shuffled_df[['department', 'quarter', 'day']] = enc.transform(shuffled_df[['department', 'quarter', 'day']])
    print(shuffled_df)

    # There is 1 column which has same fields apart from label value. This causes similarity to
    # become 0, which in turn causes divide by 0 error while calculating weighed mean, so drop it
    shuffled_df.drop_duplicates(subset=['quarter', 'department', 'day', 'team', 'smv', 'wip',
                                        'over_time', 'incentive', 'no_of_workers'], inplace=True)
    print(shuffled_df)

    # Normalize dataset for knn, note that normalization is not used for library methods
    normalized_df = shuffled_df.copy()

    # # Apply min-max normalization
    # normalized_df.loc[:, normalized_df.columns != 'actual_productivity'] = MinMaxScaler(). \
    #     fit_transform(normalized_df.loc[:, normalized_df.columns != 'actual_productivity'])
    # print(normalized_df)

    # Run knn for k values from 2 to 10
    for k_value in range(2, 11):
        print("Running created knn algorithm with euclidean distance as similarity metric and k value: ", k_value)

        # Run knn with created similarity matrix 1 and measure performance
        start_time = time.time()
        knn_1_mse, knn_1_rmse, knn_1_mape = \
            knn.measure_performance(normalized_df, k_value, number_of_folds, DistanceMetrics.EUCLIDEAN_DISTANCE)
        end_time = time.time()

        print("MSE: ", knn_1_mse)
        print("RMSE: ", knn_1_rmse)
        print("MAPE: ", knn_1_mape)
        print("Time passed: ", end_time - start_time, " seconds\n")

        print("Running created knn algorithm with manhattan distance as similarity metric and k value: ", k_value)

        # Run knn with created similarity matrix 2 and measure performance
        start_time = time.time()
        knn_2_mse, knn_2_rmse, knn_2_mape = \
            knn.measure_performance(normalized_df, k_value, number_of_folds, DistanceMetrics.MANHATTAN_DISTANCE)
        end_time = time.time()

        print("MSE: ", knn_2_mse)
        print("RMSE: ", knn_2_rmse)
        print("MAPE: ", knn_2_mape)
        print("Time passed: ", end_time - start_time, " seconds\n")

    print("Running KNeighborsRegressor with default settings")

    # Run KNeighborsRegressor from sklearn and measure performance
    start_time = time.time()
    k_neighbors_regressor_mse, k_neighbors_regressor_rmse, k_neighbors_regressor_mape = \
        k_neighbors_regressor.measure_performance(shuffled_df, number_of_folds)
    end_time = time.time()

    print("MSE: ", k_neighbors_regressor_mse)
    print("RMSE: ", k_neighbors_regressor_rmse)
    print("MAPE: ", k_neighbors_regressor_mape)
    print("Time passed: ", end_time - start_time, " seconds\n")

    print("Running DecisionTreeRegressor with default settings")

    # Run DecisionTreeRegressor from sklearn and measure performance
    start_time = time.time()
    decision_tree_regressor_mse, decision_tree_regressor_rmse, decision_tree_regressor_mape = \
        decision_tree_regressor.measure_performance(shuffled_df, number_of_folds)
    end_time = time.time()

    print("MSE: ", decision_tree_regressor_mse)
    print("RMSE: ", decision_tree_regressor_rmse)
    print("MAPE: ", decision_tree_regressor_mape)
    print("Time passed: ", end_time - start_time, " seconds\n")

    print("Running BayesianRidge regression with default settings")

    # Run BayesianRidge regression from sklearn and measure performance
    start_time = time.time()
    bayesian_ridge_mse, bayesian_ridge_rmse, bayesian_ridge_mape = \
        naive_bayes.measure_performance(shuffled_df, number_of_folds)
    end_time = time.time()

    print("MSE: ", bayesian_ridge_mse)
    print("RMSE: ", bayesian_ridge_rmse)
    print("MAPE: ", bayesian_ridge_mape)
    print("Time passed: ", end_time - start_time, " seconds\n")
