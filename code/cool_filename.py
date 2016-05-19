import os
import glob
import datetime as dt
from data import Data
import random
import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import NuSVR
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from math import isnan
from pcp import pcp



# import data set and handle missing values
data_file = "../data/prices2013.dat"
data = Data(data_file)

# take 30 historic days before the data
historic_days = 30


# choose a random date.
def get_random_day():
    days = data.get_all_days()
    rand = random.randint(historic_days, len(days))  # always have 'historic_days' previous days
    return days[rand]

def get_features_with_missing_values(features):
    features_missing_values = list()
    for feature_name, feature_list in features.items():
        for feature in feature_list:
            if isnan(feature):
                features_missing_values.append(feature_name)
                break
    return features_missing_values

act_day = get_random_day()
day = str(act_day)
# keep these stored to plot them later against the trained model
features = data.get_features_for_prev_days(dt.datetime.strptime(day, '%Y-%m-%d').date(), dt.timedelta(historic_days))
print "Features with missing values: "+str(get_features_with_missing_values(features))
flattened_features = data.flatten_features(features)
historic_data_set = data.handle_missing_values(flattened_features)

target_features = data.get_target_for_prev_days(dt.datetime.strptime(day, '%Y-%m-%d').date(),
                                                dt.timedelta(historic_days))
flattened_target_features = data.flatten_features(target_features)
target_data_set = data.handle_missing_values(flattened_target_features)

# get next week to get the actual data later to compare the predictions against
# add one extra day since I am not sure if the data form the 30th day was included in the historic set.
end_day = act_day + dt.timedelta(days=8)
future_features = data.get_features_for_prev_days(end_day, dt.timedelta(days=7))
flattened_future_features = data.flatten_features(future_features)
future_data_set = data.handle_missing_values(flattened_future_features)

future_target = data.get_target_for_prev_days(end_day, dt.timedelta(days=7))
flattened_future_target = data.flatten_features(future_target)
future_target_data_set = data.handle_missing_values(flattened_future_target)

print "Random Day = " + day
print "End of next week = " + str(end_day)

# Experiment

# RPCA Tests
L, S, (u, s, v) = pcp(target_data_set, maxiter=30, verbose=False, svd_method="approximate")
L = np.ravel(L)
S = np.ravel(S)
LD, SD, (uD, sD, vD) = pcp(historic_data_set, maxiter=30, verbose=False, svd_method="exact")


#plt.figure(3)
#plt.plot(target_data_set, label="target")
#plt.plot(L, label="low_rank")
#plt.plot(S, label="sparse")
#plt.plot(L + S, label="reconstructed")
#plt.grid(True)
#plt.legend()
#plt.show()


# train on historic data.

avg = np.median(L)
# regressorB = AdaBoostRegressor(DecisionTreeRegressor(max_depth=2), n_estimators=300)
# regressorB = LinearSVR()
regressorB = BaggingRegressor(base_estimator=DecisionTreeRegressor(max_depth=2))
# regressorB = DecisionTreeRegressor(max_depth=4)
# regressorB = RandomForestRegressor()
# regressorB = linear_model.TheilSenRegressor()
# regressorB = linear_model.Ridge()
# regressorB = linear_model.LinearRegression()
# regressorB = linear_model.PassiveAggressiveRegressor()
# regressorB = linear_model.SGDRegressor()
# regressorB = linear_model.Lasso()
# regressorB = linear_model.RANSACRegressor()
# regressorB = RadiusNeighborsRegressor(radius=1.0)
# regressorB = KNeighborsRegressor(n_neighbors=3)
regressorB.fit(historic_data_set, L)

# regressorA = AdaBoostRegressor(DecisionTreeRegressor(max_depth=2), n_estimators=300)
# regressorA = DecisionTreeRegressor(max_depth=2)
# regressorA = RandomForestRegressor()
# regressorA = BaggingRegressor(base_estimator=DecisionTreeRegressor(max_depth=2))
# regressorA = SVR(kernel='rbf', C=50, gamma=10)
regressorA = LinearSVR()
# regressorA = NuSVR(kernel='rbf', C=1e3, gamma=0.1)
# regressorA = KernelRidge(alpha=1.0, coef0=1, degree=3, gamma=None, kernel='poly', kernel_params=None)
# regressorA = linear_model.TheilSenRegressor()
# regressorA = linear_model.Ridge()
# regressorA = linear_model.BayesianRidge()
# regressorA = linear_model.LinearRegression()
# regressorA = linear_model.PassiveAggressiveRegressor()
# regressorA = linear_model.SGDRegressor()
# regressorA = linear_model.Lasso()
# regressorA = linear_model.RANSACRegressor()
# regressorA = RadiusNeighborsRegressor(radius=1.0)
# regressorA = KNeighborsRegressor(n_neighbors=4)
regressorA.fit(historic_data_set, S)

baseRegressor = linear_model.LinearRegression()
baseRegressor.fit(historic_data_set, target_data_set)

# print "Using following model: " + str(regressorA) + str(regressorB)

# plot the trained models against the data they were trained on
# together with least squares measures(in order to experiment with diff linear models)

fits_base = [regressorB.predict(historic_data_set)]
fits_base = fits_base[0]

fits_anomaly = [regressorA.predict(historic_data_set)]
fits_anomaly = fits_anomaly[0]

fits_same = [a_i + b_i for a_i, b_i in zip(fits_base, fits_anomaly)]

fits_dummy = [baseRegressor.predict(historic_data_set)]
fits_dummy = fits_dummy[0]

plt.figure(1)
plt.subplot(311)
plt.plot(fits_same, label="fits_same")
plt.plot(fits_base, label="base_fit")
plt.plot(fits_anomaly, label="anomaly_fit")
plt.plot(target_data_set, label="trgt_fnc")
plt.plot(fits_dummy, label="fits_dummy")
plt.grid(True)
plt.legend()

# plot the predicted values (by the model) against the actual prices for that week
# it is this prediction that we'll feed to the scheduler

fits_next_week_base = [regressorB.predict(future_data_set)]
fits_next_week_base = fits_next_week_base[0]


#exp
# print len(future_data_set)
# set = historic_data_set[:-len(future_data_set)]
# print len(set)
# set = np.append(set, future_data_set, axis=0)
# print len(set)

fits_next_week_anomaly = [regressorA.predict(future_data_set)]
fits_next_week_anomaly = fits_next_week_anomaly[0]

fits_next_week_dummy = [baseRegressor.predict(future_data_set)]
fits_next_week_dummy = fits_next_week_dummy[0]

fits_next_week = [a + b for a, b in zip(fits_next_week_base, fits_next_week_anomaly)]

plt.subplot(312)
plt.plot(fits_next_week, label="fits_nxt_wk")
plt.plot(fits_next_week_base, label="fits_nxt_wk_base")
plt.plot(fits_next_week_anomaly, label="fits_next_wk_anomaly")
plt.plot(future_target_data_set, label="trgt_nxt_wk")
plt.plot(fits_next_week_dummy, label="fits_nxt_wk_dummy")
plt.grid(True)
plt.legend()

errs = [(a_i - b_i)**2 for a_i, b_i in zip(fits_next_week, future_target_data_set)]
errs_base = [(a_i - b_i)**2 for a_i, b_i in zip(fits_next_week_dummy, future_target_data_set)]
# The mean square error
print("Residual sum of squares new fit: %.2f"
      % np.sum(errs))
print("Residual sum of squares old fit: %.2f"
      % np.sum(errs_base))
# Explained variance score: 1 is perfect prediction
print('Variance score base: %.2f' % regressorB.score(future_data_set, future_target_data_set))
print('Variance score anomaly: %.2f' % regressorA.score(future_data_set, future_target_data_set))
print('Variance score base fit: %.2f' % baseRegressor.score(future_data_set, future_target_data_set))

plt.subplot(313)
plt.plot(errs, label="Errors new fit")
plt.plot(errs_base, label="Errors dummy fit")
plt.grid(True)
plt.legend()
plt.show()


f_instances = "../load2"
if os.path.isdir("../load2"):
    globpatt = os.path.join("../load2", 'day*.txt')
    f_instances = sorted(glob.glob(globpatt))

# zelfde als hier boven maar de preds en actual targets zijn per day gesplitst.
preds = [] # per day an array containing a prediction for each PeriodOfDay
actuals = [] # also per day
for (i,f) in enumerate(f_instances):
    today = act_day + dt.timedelta(i)
    rows_tod = data.get_features_for_day(today)
    flattened_rows_today = data.flatten_features(rows_tod)
    X_test = data.handle_missing_values(flattened_rows_today)
    target_today = data.get_target_for_day(today)
    flattened_target_today = data.flatten_features(target_today)
    y_test = data.handle_missing_values(flattened_target_today)
    preds.append(regressorB.predict(X_test))
    actuals.append(y_test)

fig, ((ax1, ax2, ax3, ax4, ax5, ax6, ax7), (ax8, ax9, ax10, ax11, ax12, ax13, ax14)) = plt.subplots(nrows=2, ncols=7)
ax1.plot(preds[0], label="predicted")
ax1.plot(actuals[0], label="actuals")
ax2.plot(preds[1], label="predicted")
ax2.plot(actuals[1], label="actuals")
ax3.plot(preds[2], label="predicted")
ax3.plot(actuals[2], label="actuals")
ax4.plot(preds[3], label="predicted")
ax4.plot(actuals[3], label="actuals")
ax5.plot(preds[4], label="predicted")
ax5.plot(actuals[4], label="actuals")
ax6.plot(preds[5], label="predicted")
ax6.plot(actuals[5], label="actuals")
ax7.plot(preds[6], label="predicted")
ax7.plot(actuals[6], label="actuals")
ax8.plot(preds[7], label="predicted")
ax8.plot(actuals[7], label="actuals")
ax9.plot(preds[8], label="predicted")
ax9.plot(actuals[8], label="actuals")
ax10.plot(preds[9], label="predicted")
ax10.plot(actuals[9], label="actuals")
ax11.plot(preds[10], label="predicted")
ax11.plot(actuals[10], label="actuals")
ax12.plot(preds[11], label="predicted")
ax12.plot(actuals[11], label="actuals")
ax13.plot(preds[12], label="predicted")
ax13.plot(actuals[12], label="actuals")
ax14.plot(preds[13], label="predicted")
ax14.plot(actuals[13], label="actuals")
# plt.show()


# Feed the predicted data to the scheduler and calculate the cost
