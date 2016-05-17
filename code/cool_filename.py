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


act_day = get_random_day()
day = str(act_day)
# keep these stored to plot them later against the trained model
features = data.get_features_for_prev_days(dt.datetime.strptime(day, '%Y-%m-%d').date(), dt.timedelta(historic_days))
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

# train on historic data.

# regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=2), n_estimators=300)
# regressor = DecisionTreeRegressor(max_depth=2)
regressor = linear_model.TheilSenRegressor()
# regressor = linear_model.Ridge()
# regressor = linear_model.LinearRegression()
regressor.fit(historic_data_set, target_data_set)

print "Using following model: " + str(regressor)

# plot the trained models against the data they were trained on
# together with least squares measures(in order to experiment with diff linear models)

fits_same = [regressor.predict(historic_data_set)]
fits_same = fits_same[0]

plt.figure(1)
plt.subplot(311)
plt.plot(fits_same, label="fits_same")
plt.plot(target_data_set, label="trgt_fnc")
plt.grid(True)
plt.legend()

# plot the predicted values (by the model) against the actual prices for that week
# it is this prediction that we'll feed to the scheduler

fits_next_week = [regressor.predict(future_data_set)]
fits_next_week = fits_next_week[0]

plt.subplot(312)
plt.plot(fits_next_week, label="fits_nxt_wk")
plt.plot(future_target_data_set, label="trgt_nxt_wk")
plt.grid(True)
plt.legend()

errs = [(a_i - b_i)**2 for a_i, b_i in zip(fits_next_week, future_target_data_set)]
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean(errs))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regressor.score(future_data_set, future_target_data_set))

plt.subplot(313)
plt.plot(errs, label="Errors")
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
    preds.append(regressor.predict(X_test))
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


# Feed the predicted data to the scheduler and calculate the cost
