from time import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer

train_df = pd.read_csv('datasets/students_grades/train.csv',
                       names=['g0', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6'])

# Remove samples with 5 cols of zero
train_df = train_df.loc[(train_df.iloc[:, :6] != 0).any(axis=1)]

n_samples = train_df.shape[0]
n_features = train_df.shape[1]
x = train_df.iloc[:, :6]
y = train_df.iloc[:, 6]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# pipeline
# estimator = Pipeline([("imputer", Imputer(missing_values=0,
#                                           strategy="median",
#                                           axis=0)),
#                       ("gradient_boosting",
#                        GradientBoostingRegressor())])

# estimator = Pipeline([("imputer", Imputer(missing_values=0,
#                                           strategy="median",
#                                           axis=0)),
#                       ("lasso",
#                        Lasso(fit_intercept=False, max_iter=1000, normalize=False, selection='random'))])

# estimator = Pipeline([("imputer", Imputer(missing_values=0,
#                                           strategy="median",
#                                           axis=0)),
#                       ("linear_regression",
#                        LinearRegression())])

estimator = Pipeline([("imputer", Imputer(missing_values=0,
                                          strategy="median",
                                          axis=0)),
                      ("adaboost",
                       AdaBoostRegressor(loss='linear', n_estimators=5000))])

# tuning model hyper-parameters

# parameters = {
#     'gradient_boosting__loss': ('ls', 'lad', 'huber', 'quantile'),
#     'gradient_boosting__learning_rate': (0.01, 0.03, 0.1, 0.3),
#     'gradient_boosting__n_estimators': (10, 100, 1000),
#     'gradient_boosting__max_depth': (10, 100, 1000),
#     'gradient_boosting__min_samples_split': (2, 4, 8),
#     'gradient_boosting__max_features': ('auto', 'log2', 'sqrt', None),
# }

# parameters = {
#     'lasso__fit_intercept': (True, False),
#     'lasso__normalize': (True, False),
#     'lasso__max_iter': (100, 500, 1000, 5000, 10000),
#     'lasso__selection': ('cyclic', 'random'),
# }

# parameters = {
#     'linear_regression__fit_intercept': (True, False),
#     'linear_regression__normalize': (True, False),
# }

# parameters = {
#     'adaboost__n_estimators': (10, 50, 100, 500, 1000, 5000),
#     'adaboost__learning_rate': (0.01, 0.03, 0.1, 0.3, 1, 3),
#     'adaboost__loss': ('linear', 'square', 'exponential'),
#     'adaboost__random_state': (None, 1, 5, 10),
# }

# grid_search = GridSearchCV(estimator, parameters, scoring='neg_mean_squared_error')
# start_time = time()
# grid_search.fit(x, y)
# print("done in %0.3fs" % (time() - start_time))
#
# print("Best score: %0.3f" % grid_search.best_score_)
# print("Best score (sqrt): %0.3f" % np.sqrt(np.abs(grid_search.best_score_)))
# print("Best parameters set:")
# best_parameters = grid_search.best_estimator_.get_params()
# for param_name in sorted(parameters.keys()):
#     print("\t%s: %r" % (param_name, best_parameters[param_name]))

# cross validating across data with tuned model
kf = KFold(n_splits=10)
scores = cross_val_score(estimator,
                         x,
                         y,
                         cv=kf,
                         scoring=make_scorer(mean_squared_error))
print(np.sqrt(scores).mean())
