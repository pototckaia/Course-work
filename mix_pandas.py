import pandas as pd
import numpy as np


def convert_date(df, name):
    df[name] = pd.to_datetime(df[name])


def set_index_date(df, name):
    convert_date(df, name)
    df.set_index(name, inplace=True)
    df.sort_index(inplace=True)


def month_eq(df, month):
    return df[df.index.month == month]


def year_eq(df, year):
    return df[df.index.year == year]


def year_less_eq(df, year):
    return df[df.index.year <= year]


def year_great(df, year):
    return df[df.index.year > year]


def clean(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


def mean_day(df):
    x = df.groupby([df.index.year,
                   df.index.month,
                   df.index.day]).mean()

    a = pd.DataFrame(x.index.tolist(),
                     columns=['year', 'month', 'day'])
    x.index = pd.to_datetime(a)
    return x


def mean_pair(df):
    x = []
    size = df.shape[0]
    for i in range(0, size, 2):
        if i + 1 < size:
            # iloc not include end in slice
            x.append(df.iloc[i:i + 2].mean().values)
        else:
            x.append(df.iloc[i].values)
    return pd.DataFrame(x, columns=df.columns)


def diff_pair(df):
    x = []
    size = df.shape[0]
    for i in range(0, size, 2):
        if i + 1 < size:
                save = df.iloc[i:i + 2].diff(-1).fillna(0)
                save = save.iloc[0].values
                # iloc not include end in slice
                x.append(save)
        else:
            x.append(np.zeros(df.shape[1]))
    return pd.DataFrame(x, columns=df.columns)


def get_delta_day(df, date, day):
    offset_day = pd.to_timedelta(day, unit='day')
    start_date = date - offset_day
    end_date = date + offset_day
    return df.iloc[(df.index >= start_date) & (df.index <= end_date)]


def winsorized(x, name, quant):
    w = x[[name]]
    for index, row in x.iterrows():
        s = get_delta_day(x, index, 5)[[name]]
        q = s.quantile(quant)
        v = row[name]

        if v < q.iloc[0, 0] or v > q.iloc[1, 0]:
            w.drop([index], inplace=True)
    return w

    # X['winsorized'] = X[cn.value]
    # for index, row in X.iterrows():
    #     offset_day = pd.to_timedelta(5, unit='day')
    #     start_date = index - offset_day
    #     end_date = index + offset_day
    #     s = X.iloc[(X.index >= start_date) & (X.index <= end_date)]
    #     s = s[[cn.value]]
    #     q = s.quantile([0.05, 0.95])
    #     v = row[cn.value]
    #     if (v < q.iloc[0, 0] or v > q.iloc[1, 0]):
    #         X.drop([index], inplace=True)
    # print(X.shape)
    # X[['winsorized', cn.value]].plot(style='.')