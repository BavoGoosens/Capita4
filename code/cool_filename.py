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
import scipy.fftpack as ff
runcheck = __import__('mzn-runcheck')
import tempfile
import checker_mzn as chkmzn
from warnings import filterwarnings
from regressor import Regressor

filterwarnings("ignore")

# import data set and handle missing values
data_file = "../data/prices2013.dat"
data = Data(data_file)

# take historic days before the data
historic_days = 50

act_day = data.get_random_day()
act_day = dt.date(2013, 2, 1)
day = str(act_day)
# keep these stored to plot them later against the trained model
features = data.get_features_for_prev_days(dt.datetime.strptime(day, '%Y-%m-%d').date(), dt.timedelta(historic_days))
frequency = defaultdict(list)
useful = [
            'ForecastWindProduction',
            'SystemLoadEA',
            'SMPEA',
            'ORKTemperature',
            'ORKWindspeed',
            'CO2Intensity',
            'ActualWindProduction',
            'SystemLoadEP2']
for (k, v) in features.items():
    if k in useful:
        new_key = "fft_" + k
        values = ff.fft(v)
        frequency[new_key] = values
z = features.copy()
z.update(frequency)
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

frequency = defaultdict(list)
useful = [
            'ForecastWindProduction',
            'SystemLoadEA',
            'SMPEA',
            'ORKTemperature',
            'ORKWindspeed',
            'CO2Intensity',
            'ActualWindProduction',
            'SystemLoadEP2']
for (k, v) in future_features.items():
    if k in useful:
        new_key = "fft_" + k
        values = ff.fft(v)
        frequency[new_key] = values
y = future_features.copy()
y.update(frequency)


print "Start day = " + day
print "End day = " + str(end_day)

# Experiment

# RPCA Tests
L, S, (u, s, v) = pcp(np.array(target_data_set), maxiter=30, verbose=False, svd_method="approximate")
L = np.ravel(L)
S = np.ravel(S)
LD, SD, (uD, sD, vD) = pcp(np.array(historic_data_set), maxiter=30, verbose=False, svd_method="exact")


#plt.figure(3)
#plt.plot(target_data_set, label="target")
#plt.plot(L, label="low_rank")
#plt.plot(S, label="sparse")
#plt.plot(L + S, label="reconstructed")
#plt.grid(True)
#plt.legend()
#plt.show()


# train on historic data.
# regressorB = AdaBoostRegressor(DecisionTreeRegressor(max_depth=2), n_estimators=300)
# regressorB = LinearSVR()
# regressorB = BaggingRegressor(base_estimator=DecisionTreeRegressor(max_depth=2))
# regressorB = DecisionTreeRegressor(max_depth=4)
# regressorB = RandomForestRegressor()
# regressorB = linear_model.TheilSenRegressor()
# regressorB = linear_model.Ridge()
# regressorB = linear_model.LinearRegression()
# regressorB = linear_model.PassiveAggressiveRegressor()
# regressorB = linear_model.SGDRegressor()
regressorB = linear_model.Lasso()
# regressorB = linear_model.RANSACRegressor()
# regressorB = RadiusNeighborsRegressor(radius=1.0)
# regressorB = KNeighborsRegressor(n_neighbors=3)

# regressorA = AdaBoostRegressor(DecisionTreeRegressor(max_depth=2), n_estimators=300)
# regressorA = DecisionTreeRegressor(max_depth=2)
# regressorA = RandomForestRegressor()
# regressorA = BaggingRegressor(linear_model.Lasso())
# regressorA = SVR(kernel='poly', C=50, gamma=10)
# regressorA = LinearSVR()
# regressorA = GradientBoostingRegressor()
# regressorA = NuSVR(kernel='rbf', C=1e3, gamma=0.1)
# regressorA = KernelRidge(alpha=1.0, coef0=1, degree=3, gamma=None, kernel='poly', kernel_params=None)
# regressorA = linear_model.TheilSenRegressor()
regressorA = linear_model.Ridge()
# regressorA = linear_model.BayesianRidge()
# regressorA = linear_model.LinearRegression()
# regressorA = linear_model.PassiveAggressiveRegressor()
# regressorA = linear_model.SGDRegressor()
# regressorA = linear_model.Lasso()
# regressorA = linear_model.RANSACRegressor()
# regressorA = RadiusNeighborsRegressor(radius=1.0)
# regressorA = KNeighborsRegressor(n_neighbors=4)

baseRegressor = linear_model.LinearRegression()

regressor = Regressor(regressorA, regressorB, baseRegressor)
regressor.fit(historic_data_set, target_data_set)

# print "Using following model: " + str(regressorA) + str(regressorB)

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

fits_next_week_base = regressorB.predict(future_data_set)
fits_next_week_anomaly = regressorA.predict(future_data_set)
fits_next_week = [a + b for a, b in zip(fits_next_week_base, fits_next_week_anomaly)]
fits_next_week_dummy = baseRegressor.predict(future_data_set)

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
    predA = regressorA.predict(X_test)
    predB = regressorB.predict(X_test)
    pred = [x_i + y_i for x_i, y_i in zip(predA, predB)]
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