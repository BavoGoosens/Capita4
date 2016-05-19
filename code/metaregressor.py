from numpy import mean


class MetaRegressor(object):

    regressors = None

    def __init__(self, regressors):
        self.regressors = regressors

    def fit(self, X, y):
        for regressor in self.regressors:
            regressor.fit(X, y)

    def predict(self, X):
        predictions = list()
        final_prediction = list()
        for regressor in self.regressors:
            prediction = regressor.predict(X)
            predictions.append(prediction)
        for i in range(0, len(predictions[0])):
            row = list()
            for prediction in predictions:
                row.append(prediction[i])
            final_prediction.append(mean(row))
        return final_prediction

    def score(self, X, y):
        final_score = 0
        for regressor in self.regressors:
            score = regressor.score(X, y)
            final_score += score
        final_score /= len(self.regressors)
        return final_score
