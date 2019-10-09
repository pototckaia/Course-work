import numpy as np
import pandas as pdHelp

import numpy as np
import pandas as pd
import scipy.sparse
import xgboost as xgb
import mix_pandas as mix
import predict as predict_mix
import db_column_name as db

import xgboost as xgb
import mix_pandas as mix
import predict as predict_mix
import db_column_name as db

import matplotlib.pyplot as plt

from pylab import rcParams
rcParams['figure.figsize'] = 10, 10


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

# X = mix.mean_day(X)
# target_minT.index = target_minT.index.round('D')

X = X.drop([cn.offset], axis=1)

target_minT = target_minT.reindex(X.index)
target_minT = mix.clean(target_minT)
X = X.reindex(target_minT.index)
X = mix.clean(X)
print(X.shape)

target_minT = target_minT.iloc[3:] # remove on change

target_minT = mix.winsorized(target_minT, cn.value, [0.05, 0.95], 5)
X = X.reindex(target_minT.index)
print(X.shape)




from sklearn.feature_selection import SelectFromModel



params = {
    'max_depth': 3,
    'min_child_weight': 3.01,
}
reg_importances = xgb.XGBRegressor(**params)
predict = predict_mix.predict_model_split(reg_importances, X, target_minT, cn.value, 5)

importances = pd.DataFrame(reg_importances.feature_importances_, index=X.columns, columns=['Score'])
importances = importances.sort_values(by=['Score'], ascending=False)


# xgb.plot_importance(reg, importance_type='gain')

slice_importances = importances.iloc[:10]
X_select = X.loc[:, slice_importances.index]
X_select.head()


params = {
    'verbosity':0,
    'max_depth': 4,
    
    'min_child_weight': 6,
#     'learning_rate': 0.03,
}

reg = xgb.XGBRegressor(**params)

predict = predict_mix.predict_model_split(reg, X_select, target_minT, cn.value, 5)
for i, (train, test) in enumerate(predict):
    print("Train size {}".format(train.shape[0]))
    predict_mix.print_mean(train[[cn.value]], train[['prediction']])
    print("Test size {}".format(test.shape[0]))
    predict_mix.print_mean(test[[cn.value]], test[['prediction']])
    print()


    test.plot(style='.')
    plt.savefig('./im_{}.svg'.format(i))