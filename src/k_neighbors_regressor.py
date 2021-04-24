# This class uses sklearn-KNeighborsRegressor to predict productivity in
# garments_worker_productivity dataset.
import numpy
import numpy as np
import pandas
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.neighbors import KNeighborsRegressor


def measure_performance(df: pandas.DataFrame, number_of_folds: int):

    split_dfs = np.array_split(df, number_of_folds)

    mse_list = list()
    rmse_list = list()
    mape_list = list()

    for i in range(number_of_folds):
        # Get training set by appending elements other than current fold
        train_set = pandas.DataFrame()
        for j in range(number_of_folds):
            if j != i:
                train_set = train_set.append(split_dfs[j])

        test_set = split_dfs[i]

        # Default k value for KNeighborsRegressor is 5
        neigh = KNeighborsRegressor()
        x = train_set.drop(['actual_productivity'], axis=1)
        y = train_set[['actual_productivity']]
        neigh.fit(x, y)

        for index, row in test_set.iterrows():
            test_data_x = row.drop(['actual_productivity'])

            test_data_as_list = test_data_x.values.tolist()

            actual = list()
            prediction = neigh.predict(numpy.array(test_data_as_list).reshape(1, -1))[0]
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
