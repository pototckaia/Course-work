from scipy.fftpack import fft, dct
import numpy as np
import pandas as pd
import db_column_name as db


def toComplex(radii, angles):
    return radii * np.around(np.exp(1j*angles), 10)


def reshapeToComplex(r, a):
    Y_ = np.array(toComplex(r, a).values)
    Y_ = Y_.reshape((Y_.size, 1))
    return Y_


def fft_target(df, target, name, indexName):
    N = df.shape[0]
    cn = db.ColumnName()

    y = target.reset_index()
    Y = fft(y[[name]].values)
    index = np.fft.fftfreq(N)

    amplitude = np.abs(Y)
    phase = np.angle(Y)

    fft_target = pd.DataFrame(np.hstack((amplitude, phase)), columns=['amplitude', 'phase'])
    fft_target.loc[:, cn.date] = y[indexName]

    fft_target.set_index(index, inplace=True)
    fft_X = df.set_index(index)

    return fft_X, fft_target


def dct_target(df, target, name, params_dct):
    N = df.shape[0]
    cn = db.ColumnName()

    y = target.reset_index()

    Y = dct(y[[name]].values, **params_dct)
    index = np.fft.fftfreq(N)

    dct_target = pd.DataFrame(Y, columns=['dct'])
    dct_target.loc[:, cn.date] = y['index']
    dct_target.set_index(index, inplace=True)
    dct_X = df.set_index(index)

    return dct_X, dct_target