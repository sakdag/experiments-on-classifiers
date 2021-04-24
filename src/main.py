import pandas
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src import knn, k_neighbors_regressor, decision_tree_regressor

if __name__ == '__main__':

    k_value = 5
    number_of_folds = 3

    df = pd.read_csv("C:/Users/sakdag/Desktop/Ms_1.2/CEng514/HW2/experiments-on-classifiers/resources/"
                     "garments_worker_productivity.csv")
    print(df)

    # Eliminate not numeric and missing element columns for now
    df_selected = df.drop(['date', 'quarter', 'department', 'day', 'wip'], axis=1)
    print(df_selected)

    # Apply min-max normalization
    df_selected.loc[:, df_selected.columns != 'actual_productivity'] = MinMaxScaler().\
        fit_transform(df_selected.loc[:, df_selected.columns != 'actual_productivity'])
    print(df_selected)

    # Shuffle elements of dataframe
    shuffled_df = df_selected.sample(frac=1)
    print(shuffled_df)

    print("Running KNeighborsRegressor with default settings")

    # Run KNeighborsRegressor from sklearn and measure performance
    k_neighbors_regressor_mse, k_neighbors_regressor_rmse, k_neighbors_regressor_mape = \
        k_neighbors_regressor.measure_performance(shuffled_df, number_of_folds)

    print("MSE: ", k_neighbors_regressor_mse)
    print("RMSE: ", k_neighbors_regressor_rmse)
    print("MAPE: ", k_neighbors_regressor_mape, "\n")

    print("Running DecisionTreeRegressor with default settings")

    # Run DecisionTreeRegressor from sklearn and measure performance
    decision_tree_regressor_mse, decision_tree_regressor_rmse, decision_tree_regressor_mape = \
        decision_tree_regressor.measure_performance(shuffled_df, number_of_folds)

    print("MSE: ", decision_tree_regressor_mse)
    print("RMSE: ", decision_tree_regressor_rmse)
    print("MAPE: ", decision_tree_regressor_mape, "\n")

    # Run knn and measure performance
    # knn.measure_performance(rest, split_dfs[i], k_value, columns_to_use)
