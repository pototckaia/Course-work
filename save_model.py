import numpy as np
import pandas as pd
import scipy.sparse
import pickle
import xgboost as xgb

import seaborn as sns
import matplotlib.pyplot as plt

from pylab import rcParams
rcParams['figure.figsize'] = 8,5

import mix as mix
import model as pf

split_year = 2017

param_model = {'offset': 21, 'feature': 'min td850', 'day_delta': 2}

target_minT = pd.read_csv('./date/31088_103.csv')
mix.base_thing(target_minT)
X = pd.read_csv('./date/character_31088.csv')
mix.base_thing(X)

target_train = mix.year_less_eq(target_minT, split_year)
X_train = mix.year_less_eq(X, split_year)

model_train = pf.Model1(X_train, target_train, **param_model)
model_train.save('model1_train_2')

model_test = pf.Model1(X, target_minT, **param_model)

model_test.target = mix.year_great(model_test.target, split_year)
model_test.X_train = mix.year_great(model_test.X_train, split_year)

model_test.save('model1_test_2')