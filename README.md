# Experiments with Classifiers
This repository includes basic implementation of KNN and experiments with 3 regressors. In the end regression results
for 'actual_productivity' attribute using implemented KNN is compared with 3 different regressors from 
[scikit-learn](https://scikit-learn.org/stable/) repository:

1. Implementation of knn algorithm
2. Experiments with KNeighborsRegressor
3. Experiments with BayesialRidgeRegressor
4. Experiments with DecisionTreeRegressor

## Dataset

Dataset used in this project is [Productivity Prediction of Garment Employees Data Set](https://archive.ics.uci.edu/ml/datasets/Productivity+Prediction+of+Garment+Employees).
This dataset is provided by UC Irvine. Value to be predicted is 'actual_productivity'. How preprocessing is applied
can be seen in src/preprocessing/preprocessing.py.

Detailed information on KNN implementation and each regressor can be found below. For both KNN and regressors 
prediction performance is reported in terms of Mean Squared Error (MSE), Root Mean Squared Error (RMSE) and 
Mean Absolute Performance Error (MAPE). In addition, prediction time is reported. Pyhton 3.9 and Conda environment with 
dependencies as given in requirements.txt is used.

### 1. KNN implementation

This is naive implementation of KNN where index structure is not implemented. For every instance to predict, 
similarities with each training instance is calculated to find k-nearest neighbors. Weighed mean is used to
predict 'actual_productivity' field.

Implementation can be found under src/supervised_learning/knn.py. For detailed information on the command line options, 
use -h option.

### 2, 3 and 4. Experiments with Regressors

Experiments for KNeighborsRegressor, BayesialRidgeRegressor and DecisionTreeRegressor can be found under 
src/supervised_learning/k_neighbors_regressor.py, src/supervised_learning/naive_bayes.py and 
src/supervised_learning/decision_tree_regressor.py respectively. Note that these files are quite similar to each other,
however, as I use them to conduct different experiment, change parameters and try different things, duplicated parts
are not attempted to be merged. 

You can run each regressor seperately. For detailed information on the command line options use -h option.

If you want to run all experiments at the same time with default parameters, you can use src/main.py.

## License
[MIT](https://choosealicense.com/licenses/mit/)
