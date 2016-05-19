from matplotlib import pyplot as plt
import datetime as dt
from data import Data

data_file = "../data/prices2013.dat"
data = Data(data_file)
start = dt.datetime.strptime("2012-12-1", '%Y-%m-%d').date()
delta = dt.timedelta(50)
# DIT GEEFT EEN DICTIONARY VAN LIJSTEN TERUG IPV EEN LIJST VAN DICTIONARIES
# DAT IS MAKKELIJKER OM 1 FEATURE TE PLOTTEN (gewoon via key opvragen) EN OM NAN's ERUIT TE HALEN
features = data.get_features_for_prev_days(start, delta)
#features = data.get_features_for_all_days()

# DIT ZET DE DICTIONARY TERUG OM NAAR EEN LIJST VAN LIJSTEN
# DE RIJEN ZIJN DAN DE INSTANCES, DE KOLOMMEN DE FEATURES
# DIT KAN GEBRUIKT WORDEN OM TE TRAINEN
flattened_features = data.flatten_features(features)

flattened_features_without_nan = data.handle_missing_values(flattened_features)
labels = data.get_labels_for_prev_days(start, delta)

feature_data = features["SMPEA"]
feature_data2 = features["ORKTemperature"]
plt.plot(feature_data)
plt.plot(feature_data2)
plt.show()

'''feature_data1 = list()
feature_data2 = list()
feature_data3 = list()
feature_data4 = list()
feature_data5 = list()
feature_data6 = list()
feature_data7 = list()
feature_data8 = list()
feature_data9 = list()
feature_data10 = list()
for period in data_days:
    feature_data1.append(period["SMPEP2"])
    feature_data2.append(period["PeriodOfDay"])
    feature_data3.append(period["ForecastWindProduction"])
    feature_data4.append(period["SystemLoadEA"])
    feature_data5.append(period["SMPEA"])
    feature_data6.append(period["ORKTemperature"])
    feature_data7.append(period["ORKWindspeed"])
    feature_data8.append(period["CO2Intensity"])
    feature_data9.append(period["ActualWindProduction"])
    feature_data10.append(period["SystemLoadEP2"])

plt.plot(feature_data1)
plt.plot(feature_data2)
plt.xlabel("Cost")
plt.ylabel("Period of Day")
plt.show()

plt.plot(feature_data1)
plt.plot(feature_data3)
plt.xlabel("Cost")
plt.ylabel("ForeCasted wind production")
plt.show()

plt.plot(feature_data1)
plt.plot(feature_data4)
plt.xlabel("Cost")
plt.ylabel("SystemLoadEA")
plt.show()

plt.plot(feature_data1)
plt.plot(feature_data5)
plt.xlabel("Cost")
plt.ylabel("SMPEA")
plt.show()

plt.plot(feature_data1)
plt.plot(feature_data6)
plt.xlabel("Cost")
plt.ylabel("ORKTemperature")
plt.show()

plt.plot(feature_data1)
plt.plot(feature_data7)
plt.xlabel("Cost")
plt.ylabel("ORKWindspeed")
plt.show()

plt.plot(feature_data1)
plt.plot(feature_data8)
plt.xlabel("Cost")
plt.ylabel("CO2Intensity")
plt.show()

plt.plot(feature_data1)
plt.plot(feature_data9)
plt.xlabel("Cost")
plt.ylabel("ActualWindProduction")
plt.show()

plt.plot(feature_data1)
plt.plot(feature_data10)
plt.xlabel("Cost")
plt.ylabel("SystemLoadEP2")
plt.show()'''