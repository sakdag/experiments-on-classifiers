import time
from enum import Enum
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src import knn, k_neighbors_regressor, decision_tree_regressor


class DistanceMetrics(Enum):
    MANHATTAN_DISTANCE = 1
    EUCLIDEAN_DISTANCE = 2


if __name__ == '__main__':
    k_value = 5
    number_of_folds = 3

    df = pd.read_csv("C:/Users/sakdag/Desktop/Ms_1.2/CEng514/HW2/experiments-on-classifiers/resources/"
                     "garments_worker_productivity.csv")
    print(df)

    # Eliminate not numeric and missing element columns for now
    df_selected = df.drop(['date', 'targeted_productivity', 'idle_time',
                           'idle_men', 'no_of_style_change'], axis=1)
    print(df_selected)

    # TODO: Convert string valued columns to numeric values
    df_selected['department'] = pd.to_numeric(df_selected['department'], errors='coerce')
    df_selected['quarter'] = df_selected['quarter'].apply(lambda x: pd.factorize(x, sort=True))
    df_selected['day'] = df_selected['day'].apply(lambda x: pd.factorize(x, sort=True))
    print(df_selected)

    # Fill empty values in column wip with mean of wip values
    df_selected['wip'].fillna((df_selected['wip'].mean()), inplace=True)
    print(df_selected)

    # Apply min-max normalization
    df_selected.loc[:, df_selected.columns != 'actual_productivity'] = MinMaxScaler(). \
        fit_transform(df_selected.loc[:, df_selected.columns != 'actual_productivity'])
    print(df_selected)

    # Shuffle elements of dataframe
    shuffled_df = df_selected.sample(frac=1).reset_index(drop=True)
    print(shuffled_df)

    print("Running created knn algorithm with created similarity metric 1")

    # Run knn with created similarity matrix 1 and measure performance
    start_time = time.time()
    knn_1_mse, knn_1_rmse, knn_1_mape = \
        knn.measure_performance(shuffled_df, k_value, number_of_folds, DistanceMetrics.EUCLIDEAN_DISTANCE)
    end_time = time.time()

    print("MSE: ", knn_1_mse)
    print("RMSE: ", knn_1_rmse)
    print("MAPE: ", knn_1_mape)
    print("Time passed: ", end_time - start_time, " seconds\n")

    print("Running created knn algorithm with created similarity metric 2")

    # Run knn with created similarity matrix 2 and measure performance
    start_time = time.time()
    knn_2_mse, knn_2_rmse, knn_2_mape = \
        knn.measure_performance(shuffled_df, k_value, number_of_folds, DistanceMetrics.MANHATTAN_DISTANCE)
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
    print("Time passed: ", end_time - start_time, " seconds")

    print("Running DecisionTreeRegressor with default settings\n")

    # Run DecisionTreeRegressor from sklearn and measure performance
    start_time = time.time()
    decision_tree_regressor_mse, decision_tree_regressor_rmse, decision_tree_regressor_mape = \
        decision_tree_regressor.measure_performance(shuffled_df, number_of_folds)
    end_time = time.time()

    print("MSE: ", decision_tree_regressor_mse)
    print("RMSE: ", decision_tree_regressor_rmse)
    print("MAPE: ", decision_tree_regressor_mape)
    print("Time passed: ", end_time - start_time, " seconds\n")
