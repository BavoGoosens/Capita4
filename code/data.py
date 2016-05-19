import prices_data as pd
from collections import defaultdict
from sklearn.preprocessing import Imputer
from math import isnan


class Data(object):
    path = None
    data = None
    feature_columns = None
    predict_column = None

    def __init__(self, path):
        self.path = path
        self.data = pd.load_prices(path)
        self.feature_columns = [
            'HolidayFlag',
            'DayOfWeek',
            'PeriodOfDay',
            'ForecastWindProduction',
            'SystemLoadEA',
            'SMPEA',
            'ORKTemperature',
            'ORKWindspeed',
            'CO2Intensity',
            'ActualWindProduction',
            'SystemLoadEP2'
        ]
        self.predict_column = 'SMPEP2'

    # Returns dictionary with feature names (keys) and list of values for all instances
    # This dictionary can be used for filling in NaN's and for plotting features
    def get_features_for_day(self, day):
        result = defaultdict()
        data_day = pd.get_data_day(self.data, day)  # returns all rows for one day
        for data_row in data_day:
            for key, value in data_row.iteritems():
                if key in self.feature_columns:
                    if key not in result:
                        result[key] = list()
                    value = self.convert_type(value)
                    result[key].append(value)
        return result

    def get_target_for_day(self, day):
        result = defaultdict()
        data_day = pd.get_data_day(self.data, day)  # returns all rows for one day
        for data_row in data_day:
            for key, value in data_row.iteritems():
                if key in self.predict_column:
                    if key not in result:
                        result[key] = list()
                    value = self.convert_type(value)
                    result[key].append(value)
        return result

    # Returns dictionary with feature names (keys) and list of values for all instances
    # This dictionary can be used for filling in NaN's and for plotting features
    def get_features_for_prev_days(self, start, delta):
        result = defaultdict()
        data_days = pd.get_data_prevdays(self.data, start, delta)
        for data_row in data_days:
            for key, value in data_row.iteritems():
                if key in self.feature_columns:
                    if key not in result:
                        result[key] = list()
                    value = self.convert_type(value)
                    result[key].append(value)
        return result

    # Returns dictionary with feature names (keys) and list of values for all instances
    # This dictionary can be used for filling in NaN's and for plotting features
    def get_features_for_all_days(self):
        result = defaultdict()
        days = pd.get_all_days(self.data)  # returns all days (without data) in data set
        for day in days:
            features = self.get_features_for_day(day)
            for key, value in features.iteritems():
                if key not in result:
                    result[key] = list()
                result[key] += value
        return result

    def get_labels_for_prev_days(self, start, delta):
        result = list()
        data_days = pd.get_data_prevdays(self.data, start, delta)
        for data_row in data_days:
            for key, value in data_row.iteritems():
                if key == self.predict_column:
                    value = self.convert_type(value)
                    result.append(value)
        return result

    def get_labels_for_day(self, day):
        result = list()
        data_day = pd.get_data_day(self.data, day)  # returns all rows for one day
        for data_row in data_day:
            for key, value in data_row.iteritems():
                if key == self.predict_column:
                    value = self.convert_type(value)
                    result.append(value)
        return result

    def get_labels_for_all_days(self):
        result = list()
        days = pd.get_all_days(self.data)
        for day in days:
            labels_for_day = self.get_labels_for_day(day)
            result += labels_for_day
        return result

    def get_all_days(self):
        days = set()
        for row in self.data:
            days.add(row['datetime'].date())
        return sorted(days)

    def handle_missing_values(self, flattened_features):
        imputer = Imputer(missing_values=float('nan'), strategy='most_frequent')
        flattened_features = imputer.fit_transform(flattened_features)
        return flattened_features

    def handle_missing_values_advanced(self, features):
        features_with_missing_values = self.get_features_with_missing_values()

    def get_features_with_missing_values(self, features):
        features_missing_values = list()
        for feature_name, feature_list in features.items():
            for feature in feature_list:
                if isnan(feature):
                    features_missing_values.append(feature_name)
                    break
        return features_missing_values

    # Return the given string in the right type
    def convert_type(self, value):
        if pd.is_int(value):
            return int(value)
        if pd.is_float(value):
            return float(value)
        elif pd.is_nan(value):
            return float('nan')
        return value

    # Retransform dictionary in matrix. List (instances) of lists (values of features)
    # This matrix can be used for training.
    def flatten_features(self, features):
        result = list()
        length = len(features[features.keys()[0]])
        for i in range(0, length):
            row = list()
            for value in features.values():
                row.append(value[i])
            result.append(row)
        return result

    def get_target_for_prev_days(self, start, delta):
        result = defaultdict()
        data_days = pd.get_data_prevdays(self.data, start, delta)
        for data_row in data_days:
            for key, value in data_row.iteritems():
                if key in self.predict_column:
                    if key not in result:
                        result[key] = list()
                    value = self.convert_type(value)
                    result[key].append(value)
        return result
