print(__doc__)


import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from data import Data

from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

logistic = linear_model.ElasticNet()

pca = decomposition.PCA()
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

data_file = "../data/prices2013.dat"
data = Data(data_file)

# DIT GEEFT EEN DICTIONARY VAN LIJSTEN TERUG IPV EEN LIJST VAN DICTIONARIES
# DAT IS MAKKELIJKER OM 1 FEATURE TE PLOTTEN (gewoon via key opvragen) EN OM NAN's ERUIT TE HALEN
features = data.get_features_for_prev_days(dt.datetime.strptime("2012-12-1", '%Y-%m-%d').date(), dt.timedelta(50))
#features = data.get_features_for_all_days()

# DIT ZET DE DICTIONARY TERUG OM NAAR EEN LIJST VAN LIJSTEN
# DE RIJEN ZIJN DAN DE INSTANCES, DE KOLOMMEN DE FEATURES
# DIT KAN GEBRUIKT WORDEN OM TE TRAINEN
flattened_features = data.flatten_features(features)

flattened_features_without_nan = data.handle_missing_values(flattened_features)


# digits = datasets.load_digits()
X_digits = flattened_features_without_nan[:, 0:4]
X_digits = np.array(X_digits).astype(float)
# print X_digits
y_digits = flattened_features_without_nan[:, 5]
y_digits = np.array(y_digits).astype(float)
# digits = datasets.load_digits()
# X_digits = digits.data
print len(X_digits)
# y_digits = digits.target
print len(y_digits)


logistic.fit(X_digits, y_digits)






###############################################################################
# Plot the PCA spectrum
# pca.fit(X_digits)
#
# plt.figure(1, figsize=(4, 3))
# plt.clf()
# plt.axes([.2, .2, .7, .7])
# plt.plot(pca.explained_variance_, linewidth=2)
# plt.axis('tight')
# plt.xlabel('n_components')
# plt.ylabel('explained_variance_')

###############################################################################
# Prediction

# n_components = [4]
# Cs = np.logspace(-4, 4, 3)
#
# estimator = GridSearchCV(pipe,
#                          dict(pca__n_components=n_components,
#                               logistic__C=Cs))
# estimator.fit(X_digits, y_digits)
#
# plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
#             linestyle=':', label='n_components chosen')
# plt.legend(prop=dict(size=12))
# plt.show()