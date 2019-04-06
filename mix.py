import pandas as pd
import numpy as np
import db_column_name as db
import calendar
from datetime import timedelta
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, median_absolute_error

def convert_date(df, name):
    df[name] = pd.to_datetime(df[name])


def set_index_date(df, name):
    convert_date(df, name)
    df.set_index(name, inplace=True)


def month_eq(df, month):
    return df[df.index.month == month]


def year_eq(df, year):
    return df[df.index.year == year]


def year_less_eq(df, year):
    return df[df.index.year <= year]


def year_great(df, year):
    return df[df.index.year > year]


# +- n day with any year
def offset_n_day(df, date_point, day_delta):
    offset_day = pd.to_timedelta(day_delta, unit='day')
    start_date = date_point - offset_day
    end_date = date_point + offset_day
    return df[(df.index.month >= start_date.month &
               df.index.day >= start_date.day & df.index.hour >= start_date.hour)&
              (df.index.month <= end_date.month &
               df.index.day <= end_date.day & df.index.hour <= end_date.hour)]


def dayInYear(date):
    return 366 if calendar.isleap(date.year) else 365


def group_by_n_day(df, date, day_delta):
    offset = offset_n_day(df, date, day_delta)
    # SettingWithCopyWarning
    offset.loc[:, 'date'] = (date - offset.index) % timedelta(days=dayInYear(date))
    offset.loc[offset.index < date, 'date'].apply(lambda df: -df)
    offset = offset.groupby('date').mean().T
    offset.index = [date]
    return offset


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


def getTarget(X ):
    c = db.ColumnName()
    y = X[[c.value]]
    X = X.drop(c.value, axis=1)
    return X, y  


def print_mean(y_t, y_tr, first, second):
    mse_test = mean_squared_error(y_t[first], y_t[second])
    mse_train = mean_squared_error(y_tr[first], y_tr[second])
    print("Mean squared error on train {} and test {}".format(mse_train, mse_test))
    
    mae_test = mean_absolute_error(y_t[first], y_t[second])
    mae_train = mean_absolute_error(y_tr[first], y_tr[second])
    print("Mean absolute error on train {} and test {}".format(mae_train, mae_test))

    evs_test = explained_variance_score(y_true=y_t[first], y_pred=y_t[second])
    evs_train = explained_variance_score(y_true=y_tr[first], y_pred=y_tr[second])
    print("Explained variance score on train {} and test {}".format(evs_train, evs_test))

    r2_train = r2_score(y_true=y_tr[first], y_pred=y_tr[second])
    r2_test = r2_score(y_true=y_t[first], y_pred=y_t[second])
    print("Coefficient of determination on train {} and test {}".format(r2_train, r2_test))

    mear_train = median_absolute_error(y_true=y_tr[first], y_pred=y_tr[second])
    mear_test = median_absolute_error(y_true=y_t[first], y_pred=y_t[second])
    print("Median absolute error on train {} and test {}".format(r2_train, r2_test))

