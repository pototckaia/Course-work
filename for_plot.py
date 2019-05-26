import numpy as np
import pandas as pd
import scipy.sparse
import xgboost as xgb
import mix_pandas as mix
import predict as predict_mix
import db_column_name as db
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 10, 7


cn = db.ColumnName()

target_minT = pd.read_csv('./data/31286_103.csv')
mix.set_index_date(target_minT, cn.date)

X = pd.read_csv('./data/character_31286.csv')
mix.set_index_date(X, cn.date)

X = X.drop([cn.point], axis=1)
X = X[[x for x in X.columns if 'avg' in x or 
       x == cn.offset]]

X = X[X[cn.offset] == 69]
X = X[X.index.hour == 21]
print(X.shape)

X = X.drop([cn.offset], axis=1)

X = X.iloc[3:] # remove on change

target_minT = target_minT.reindex(X.index)
target_minT = mix.clean(target_minT)
X = X.reindex(target_minT.index)
X = mix.clean(X)

target_minT = mix.winsorized(target_minT, cn.value, [0.05, 0.95], 5)
X = X.reindex(target_minT.index)

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression, Lasso
from stepwise import stepwise_selection

selectColumns = stepwise_selection(X, target_minT)
X_select = X.loc[:, selectColumns]

X_select.head()

predict = predict_mix.predict_model_split(LinearRegression(), X_select, target_minT, cn.value, 5)
for i, (train, test) in enumerate(predict):
    predict_mix.print_mean(train[[cn.value]], train[['prediction']])
    predict_mix.print_mean(test[[cn.value]], test[['prediction']])
    test.plot(style='.')
    plt.savefig('./plot/plot/stepwise_regression_5_cv_{}_31286.svg'.format(i))