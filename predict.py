import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, median_absolute_error
from sklearn.model_selection import TimeSeriesSplit


def mean_absolute_percentage_error(y_true, y_pred):
	mask = y_true != 0
	return (np.fabs((y_true - y_pred) / y_true))[mask].mean()


def xgb_mape(preds, dtrain):
	labels = dtrain.get_label()
	return ('mape', -np.mean(np.abs((labels - preds) / (labels + 1))))


def MASE(training_series, testing_series, prediction_series):
	"""
	Computes the MEAN-ABSOLUTE SCALED ERROR forcast error for univariate time series prediction.
	parameters:
		training_series: the series used to train the model, 1d numpy array
		testing_series: the test series to predict, 1d numpy array or float
		prediction_series: the prediction of testing_series, 1d numpy array (same size as testing_series) or float
		absolute: "squares" to use sum of squares and root the result, "absolute" to use absolute values.
	"""
	n = training_series.shape[0]
	d = np.abs(np.diff(training_series)).sum() / (n - 1)
	errors = np.abs(testing_series - prediction_series)
	return errors.mean() / d


def print_mean(y_true, y_predict):
	pd.options.mode.chained_assignment = None
	mse = mean_squared_error(y_true, y_predict)
	print("Mean squared error {:.5f}".format(mse))

	mae = mean_absolute_error(y_true, y_predict)
	print("Mean absolute error {:.5f}".format(mae))

	mear = median_absolute_error(y_true, y_predict)
	print("Median absolute error {:.5f}".format(mear))

	# evs_test = explained_variance_score(y_true=y_t[first], y_pred=y_t[second])
	# evs_train = explained_variance_score(y_true=y_tr[first], y_pred=y_tr[second])
	# print("Explained variance score on train {:.5f} and test {:.5f}".format(evs_train, evs_test))

	# r2_train = r2_score(y_true=y_tr[first], y_pred=y_tr[second])
	# r2_test = r2_score(y_true=y_t[first], y_pred=y_t[second])
	# print("Coefficient of determination on train {:.5f} and test {:.5f}".format(r2_train, r2_test))


def predict_model(model, X_train, X_test, target_train, target_test, target_name):
	model.fit(X_train, target_train[[target_name]])

	target_test.loc[:, 'prediction'] = model.predict(X_test)
	target_train.loc[:, 'prediction'] = model.predict(X_train)

	return target_train, target_test


def predict_model_split(model, X, target, target_name, split):
	res = []
	for train_index, test_index in TimeSeriesSplit(split).split(X):
		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		target_train, target_test = target.iloc[train_index], target.iloc[test_index]

		model.fit(X_train, target_train[[target_name]])
		target_test.loc[:, 'prediction'] = model.predict(X_test)
		target_train.loc[:, 'prediction'] = model.predict(X_train)

		res.append((target_train, target_test))
	return res

