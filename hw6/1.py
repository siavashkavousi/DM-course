import numpy as np
from scipy.io import loadmat
from sklearn.covariance import EmpiricalCovariance, EllipticEnvelope
from sklearn.metrics import accuracy_score, classification_report

cardio_data = loadmat('cardio.mat')
estimator = EmpiricalCovariance()
cov = estimator.fit(cardio_data['X'])
mahal_cov = cov.mahalanobis(cardio_data['X'])
# sort values and extract n maximum values
# number of outliers in cardio data = 176
indexes = np.argpartition(mahal_cov, 176)[-176:]
y_pred = np.zeros(cardio_data['y'].shape)
y_pred[indexes] = 1
print(classification_report(cardio_data['y'], y_pred))
print(accuracy_score(cardio_data['y'], y_pred))

cov = EllipticEnvelope().fit(np.dot(cardio_data['X'].T, cardio_data['X']))
mahal_cov = cov.mahalanobis(cardio_data['X'])

indexes = np.argpartition(mahal_cov, 176)[-176:]
y_pred = np.zeros(cardio_data['y'].shape)
y_pred[indexes] = 1
print(classification_report(cardio_data['y'], y_pred))
print(accuracy_score(cardio_data['y'], y_pred))