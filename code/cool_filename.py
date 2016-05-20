import os
import glob
import datetime as dt
from data import Data
import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import NuSVR
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from math import isnan
from pcp import pcp
from collections import defaultdict
runcheck = __import__('mzn-runcheck')
import tempfile
import checker_mzn as chkmzn
from warnings import filterwarnings
from regressor import Regressor
from metaregressor import MetaRegressor

filterwarnings("ignore")

# import data set and handle missing values
data_file = "../data/prices2013.dat"
data = Data(data_file)

# take historic days before the data
historic_days = 50


# initialize start day
act_day = data.get_random_day()
act_day = dt.date(2013, 2, 1) # comment if day needs to be chosen randomly
day = str(act_day)

# keep these stored to plot them later against the trained model
features = data.get_features_for_prev_days(dt.datetime.strptime(day, '%Y-%m-%d').date(), dt.timedelta(historic_days))
historic_data_set = data.flatten_features(features) # for training

target_features = data.get_target_for_prev_days(dt.datetime.strptime(day, '%Y-%m-%d').date(),
                                                dt.timedelta(historic_days))
target_data_set = data.flatten_features(target_features) # for training

# get next week to get the actual data later to compare the predictions against
# add one extra day since I am not sure if the data form the 30th day was included in the historic set.
end_day = act_day + dt.timedelta(days=15)

future_features = data.get_features_for_prev_days(end_day, dt.timedelta(days=14))
future_data_set = data.flatten_features(future_features) # for predicting

future_target = data.get_target_for_prev_days(end_day, dt.timedelta(days=14))
future_target_data_set = data.flatten_features(future_target) # for testing

print "Start day = " + day
print "End day = " + str(end_day)

# Experiment
# RPCA Tests
L, S, (u, s, v) = pcp(np.array(target_data_set), maxiter=30, verbose=False, svd_method="approximate")
L = np.ravel(L)
S = np.ravel(S)
LD, SD, (uD, sD, vD) = pcp(np.array(historic_data_set), maxiter=30, verbose=False, svd_method="exact")

# plt.figure(5)
# plt.plot([item[3] for item in historic_data_set], label="historic_data_set")
# plt.plot([item[3] for item in LD], label="low_dimension")
# plt.plot([item[3] for item in SD], label="sparse")
# plt.legend()
# plt.show()


# plt.figure(3)
# plt.plot(target_data_set, label="target")
# plt.plot(L, label="low_rank")
# plt.plot(S, label="sparse")
# plt.plot(L + S, label="reconstructed")
# plt.grid(True)
# plt.legend()
# plt.show()

# plt.figure(4)
# plt.plot(future_target_data_set, label="target")
# plt.grid(True)
# plt.legend()
# plt.show()

# train on historic data.
# regressorB1 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=2), n_estimators=300)
# regressorB = LinearSVR()
# regressorB = BaggingRegressor(base_estimator=DecisionTreeRegressor(max_depth=2))
# regressorB = DecisionTreeRegressor(max_depth=4)
# regressorB = RandomForestRegressor()
# regressorB2 = linear_model.TheilSenRegressor()
# regressorB = linear_model.Ridge()
# regressorB = linear_model.LinearRegression()
# regressorB3 = linear_model.PassiveAggressiveRegressor()
# regressorB = linear_model.SGDRegressor()
regressorB2 = linear_model.Lasso()
# regressorB = linear_model.RANSACRegressor()
# regressorB = RadiusNeighborsRegressor(radius=1.0)
# regressorB = KNeighborsRegressor(n_neighbors=3)

# regressorA = AdaBoostRegressor(DecisionTreeRegressor(max_depth=2), n_estimators=300)
# regressorA = DecisionTreeRegressor(max_depth=2)
# regressorA = RandomForestRegressor()
regressorA1 = BaggingRegressor(linear_model.Lasso())
# regressorA = SVR(kernel='poly', C=50, gamma=10)
# regressorA = LinearSVR()
# regressorA = GradientBoostingRegressor()
regressorA2 = NuSVR(kernel='rbf', C=1e3, gamma=0.1)
# regressorA = KernelRidge(alpha=1.0, coef0=1, degree=3, gamma=None, kernel='poly', kernel_params=None)
# regressorA = linear_model.TheilSenRegressor()
regressorA3 = linear_model.Ridge()
# regressorA = linear_model.BayesianRidge()
regressorA4 = linear_model.LinearRegression()
regressorA5 = linear_model.PassiveAggressiveRegressor()
# regressorA = linear_model.SGDRegressor()
regressorA = linear_model.Lasso()
# regressorA = linear_model.RANSACRegressor()
# regressorA = RadiusNeighborsRegressor(radius=1.0)
# regressorA = KNeighborsRegressor(n_neighbors=4)

regressorB = MetaRegressor([regressorB2])
regressorA = MetaRegressor([regressorA1, regressorA2, regressorA3, regressorA4, regressorA5])
baseRegressor = linear_model.LinearRegression()

regressor = Regressor(regressorA, regressorB, baseRegressor)
regressor.fit(historic_data_set, target_data_set)

# plot the trained models against the data they were trained on
# together with least squares measures(in order to experiment with diff linear models)

predict_base, predict_anomaly, predict_total, predict_dummy = regressor.predict(historic_data_set)

plt.figure(1)
plt.subplot(311)
plt.plot(predict_total, label="total")
plt.plot(predict_base, label="base")
plt.plot(predict_anomaly, label="anomaly")
plt.plot(target_data_set, label="target")
plt.plot(predict_dummy, label="dummy")
plt.grid(True)
plt.legend()

# plot the predicted values (by the model) against the actual prices for that week
# it is this prediction that we'll feed to the scheduler

#exp
# print len(future_data_set)
# set = historic_data_set[:-len(future_data_set)]
# print len(set)
# set = np.append(set, future_data_set, axis=0)
# print len(set)

next_week_base, next_week_anomaly, next_week_total, next_week_dummy = regressor.predict(future_data_set)

plt.subplot(312)
plt.plot(next_week_total, label="total")
plt.plot(next_week_base, label="base")
plt.plot(next_week_anomaly, label="anomaly")
plt.plot(future_target_data_set, label="target")
plt.plot(next_week_dummy, label="dummy")
plt.grid(True)
plt.legend()

errs = regressor.get_errors(next_week_total, future_target_data_set)
errs_base = regressor.get_errors(next_week_dummy, future_target_data_set)
# The mean square error
print("Residual sum of squares new fit: %.2f"
      % np.sum(errs))
print("Residual sum of squares old fit: %.2f"
      % np.sum(errs_base))
print("AVG sum of squares new fit: %.2f"
      % np.mean(errs))
print("AVG sum of squares old fit: %.2f"
      % np.mean(errs_base))

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

f_instances = "../load1"
if os.path.isdir(f_instances):
    globpatt = os.path.join(f_instances, 'day*.txt')
    f_instances = sorted(glob.glob(globpatt))

# zelfde als hier boven maar de preds en actual targets zijn per day gesplitst.
preds = [] # per day an array containing a prediction for each PeriodOfDay
actuals = [] # also per day
for (i,f) in enumerate(f_instances):
    today = act_day + dt.timedelta(i)
    rows_tod = data.get_features_for_day(today)
    X_test = data.flatten_features(rows_tod)
    target_today = data.get_target_for_day(today)
    y_test = data.flatten_features(target_today)
    y_test = list(np.ravel(y_test))
    predA, predB, pred_total, pred_dummy = regressor.predict(X_test)
    preds.append(pred_total)
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
plt.show()

file_mzn = "../energychallenge-BavoGoosensMichielVandendriesshe.mzn"
tmpdir = tempfile.mkdtemp()
mzn_dir = "/home/michielvandendriessche/Desktop/minizinc/minizinc"
print_output = False
print_pretty = False
v = 1
# Feed the predicted data to the scheduler and calculate the cost
tot_act = 0
tot_time = 0
for i, f in enumerate(f_instances):
    #f = f.replace("../", "")
    data_forecasts = preds[i]
    data_actual = actuals[i]
    (timing, out) = runcheck.mzn_run(file_mzn, f, data_forecasts,
                                     tmpdir, mzn_dir=mzn_dir,
                                     print_output=print_output,
                                     verbose=(v - 1))
    instance = runcheck.mzn_toInstance(f, out, data_forecasts,
                                       data_actual=data_actual,
                                       pretty_print=print_pretty,
                                       verbose=(v - 1))
    if v >= 1:
        # csv print:
        if i == 0:
            # an ugly hack, print more suited header here
            print "scheduling_scenario; date; cost_forecast; cost_actual; runtime"
        today = act_day + dt.timedelta(i)
        chkmzn.print_instance_csv(f, today.__str__(), instance, timing=timing, header=False)
    instance.compute_costs()
    tot_act += instance.day.cj_act
    tot_time += timing

print ""
print "Total cost: "+str(tot_act)
print "Total time: "+str(timing)
