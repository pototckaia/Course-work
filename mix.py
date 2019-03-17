import pandas as pd
import db_column_name
import calendar
from datetime import timedelta


def convert_date(df, name):
    df[name] = pd.to_datetime(df[name])


def set_index_date(df, name):
    convert_date(df, name)
    df.set_index(name, inplace=True)

def mdHMS(date):
    return date.strftime('%m-%d %H:%M:%S')


def month_eq(df, month):
    return df[df.index.month == month]


def year_eq(df, year):
    return df[df.index.year == year]


def year_less_eq(df, year):
    return df[df.index.year <= year]


def year_great(df, year):
    return df[df.index.year > year]


# todo rewrite
# +- n day with any year
def offset_n_day(df, date_point, day_delta):
    offset_day = pd.to_timedelta(day_delta, unit='day')
    start_date = mdHMS(date_point - offset_day)
    end_date = mdHMS(date_point + offset_day)
    return df[(mdHMS(df.index) >= start_date)&(mdHMS(df.index) <= end_date)]


def map(dt, date, day_delta):
    if date - dt > timedelta(days=day_delta):
        if dt.month == 2 and dt.date == 29:
            return dt.replace(year=date.year, month=3, date=1)
        else:
            return dt.replace(year=date.year)
    return dt


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

