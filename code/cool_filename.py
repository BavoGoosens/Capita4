import os
import glob
import datetime as dt
from data import Data
import random
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
# regressor = linear_model.TheilSenRegressor()
regressor = linear_model.Ridge()
# regressor = linear_model.LinearRegression()
regressor.fit(historic_data_set, target_data_set)

print "Using following model: " + str(regressor)

# plot the trained models against the data they were trained on
# together with least squares measures(in order to experiment with diff linear models)

fits_same = [regressor.predict(historic_data_set)]
fits_same = fits_same[0]

plt.figure(1)
plt.subplot(211)
plt.plot(fits_same, label="fits_same")
plt.plot(target_data_set, label="trgt_fnc")
plt.grid(True)
plt.legend()

# plot the predicted values (by the model) against the actual prices for that week
# it is this prediction that we'll feed to the scheduler

fits_next_week = [regressor.predict(future_data_set)]
fits_next_week = fits_next_week[0]

plt.subplot(212)
plt.plot(fits_next_week, label="fits_nxt_wk")
plt.plot(future_target_data_set, label="trgt_nxt_wk")
plt.grid(True)
plt.legend()
plt.show()

f_instances = "../load2"
if os.path.isdir("../load2"):
    globpatt = os.path.join("../load2", 'day*.txt')
    f_instances = sorted(glob.glob(globpatt))

print "test "


# Feed the predicted data to the scheduler and calculate the cost
