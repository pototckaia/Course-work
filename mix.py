import pandas as pd
import db_column_name
import calendar
from datetime import timedelta


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

