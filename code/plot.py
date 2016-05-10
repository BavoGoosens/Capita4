from matplotlib import pyplot as plt
import datetime as dt
from data import Data

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

feature_data = features["SMPEA"]
feature_data2 = features["PeriodOfDay"]
plt.plot(feature_data)
plt.plot(feature_data2)
plt.show()
