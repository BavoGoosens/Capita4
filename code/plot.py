from matplotlib import pyplot as plt
import prices_data as pd
import datetime as dt
data_file = "../data/prices2013.dat"
dat = pd.load_prices(data_file)
data_days = pd.get_data_prevdays(dat, dt.datetime.strptime("2013-12-1", '%Y-%m-%d').date(), dt.timedelta(50))
feature_data = list()
feature_data2 = list()
for period in data_days:
    feature_data.append(period["SMPEP2"])
    feature_data2.append(period["PeriodOfDay"])

plt.plot(feature_data)
plt.plot(feature_data2)
plt.show()