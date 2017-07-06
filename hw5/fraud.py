import pandas as pd
from sklearn.metrics import classification_report, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv('datasets/data_fraud/X_train.csv').sort_index(axis=1)
label = pd.read_csv('datasets/data_fraud/Y_train.csv')
test_df = pd.read_csv('datasets/data_fraud/X_test.csv').sort_index(axis=1)

combined_df = train_df.append(test_df)
combined_df.reset_index(inplace=True)
combined_df.drop('index', inplace=True, axis=1)
# impute missing value with the most frequent state
combined_df['state'] = combined_df['state'].fillna('CA')


def process_state(combined):
    state_dummies = pd.get_dummies(combined['state'], prefix="state")
    combined = pd.concat([combined, state_dummies], axis=1)
    # removing "Pclass"
    combined.drop('state', axis=1, inplace=True)
    return combined


combined_df = process_state(combined_df)

combined_df.drop(['customerAttr_b', 'amount', 'hour_b'], axis=1, inplace=True)


def recover_train_test(combined):
    train = combined.ix[0:train_df.shape[0] - 1]
    test = combined.ix[train_df.shape[0]:]

    return train, test


train_df, test_df = recover_train_test(combined_df)


def describe(target_true, target_pred):
    report = classification_report(target_true, target_pred)
    print()
    print('classification report')
    print(report)
    precision = precision_score(target_true, target_pred)
    print('precision score')
    print(precision)
    recall = recall_score(target_true, target_pred)
    print('recall score')
    print(recall)


X_train, X_test, y_train, y_test = train_test_split(train_df, label, test_size=0.3)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# print('Starting decision tree classifier grid search...')
# clf = DecisionTreeClassifier()
#
# parameter_grid = {
#     'max_depth': [10, 15, 20, 50, 100],
#     'max_features': ['auto', 'sqrt', 'log2', None],
#     'class_weight': [{0: 1., 1: 2}, {0: 1., 1: 4}, {0: 1., 1: 8}, 'balanced', None],
#     'criterion': ['gini', 'entropy'],
#     'splitter': ['best', 'random'],
# }
#
# cross_validation = StratifiedKFold(n_splits=5)
#
# grid_search = GridSearchCV(clf,
#                            param_grid=parameter_grid,
#                            cv=cross_validation,
#                            scoring='f1',
#                            n_jobs=-1,
#                            verbose=1)
#
# grid_search.fit(train_df, label.values.ravel())
#
# print('Best score: {}'.format(grid_search.best_score_))
# print('Best parameters: {}'.format(grid_search.best_params_))

# print('Starting logistic regression classifier grid search...')
# clf = LogisticRegression()
#
# parameter_grid = {
#     'class_weight': [{0: 1., 1: 2}, {0: 1., 1: 4}, {0: 1., 1: 8}, 'balanced', None],
#     'max_iter': [100, 200, 500, 800, 1000],
#     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
# }
#
# cross_validation = StratifiedKFold(n_splits=5)
#
# grid_search = GridSearchCV(clf,
#                            param_grid=parameter_grid,
#                            cv=cross_validation,
#                            scoring='f1',
#                            n_jobs=-1,
#                            verbose=1)
#
# grid_search.fit(train_df, label.values.ravel())
#
# print('Best score: {}'.format(grid_search.best_score_))
# print('Best parameters: {}'.format(grid_search.best_params_))

# print('Starting random forest classifier grid search...')
# clf = RandomForestClassifier()
#
# parameter_grid = {
#     'max_depth': [10, 15, 20, 50, 100],
#     'max_features': ['auto', 'sqrt', 'log2', None],
#     'n_estimators': [10, 100, 200, 400],
#     'criterion': ['gini', 'entropy'],
#     'class_weight': [{0: 1., 1: 2}, {0: 1., 1: 4}, {0: 1., 1: 8}, 'balanced', None],
# }
#
# cross_validation = StratifiedKFold(n_splits=5)
#
# grid_search = GridSearchCV(clf,
#                            param_grid=parameter_grid,
#                            cv=cross_validation,
#                            scoring='f1',
#                            n_jobs=-1,
#                            verbose=1)
#
# grid_search.fit(train_df, label.values.ravel())
#
# print('Best score: {}'.format(grid_search.best_score_))
# print('Best parameters: {}'.format(grid_search.best_params_))

print('Starting MLP classifier grid search...')
clf = MLPClassifier(verbose=1)

parameter_grid = {
    'activation': ['logistic', 'tanh', 'relu'],
    'alpha': [0.0001, 0.0003, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [200, 500, 1000]
}

cross_validation = StratifiedKFold(n_splits=5)

grid_search = GridSearchCV(clf,
                           param_grid=parameter_grid,
                           cv=cross_validation,
                           scoring='f1',
                           n_jobs=-1,
                           verbose=1)

grid_search.fit(X_train, y_train.values.ravel())

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))
