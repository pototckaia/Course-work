import pandas as pd

import mix as mix
import model as pf
import db_column_name as db

split_year = 2017

param_model = {'offset': 69}

cn = db.ColumnName()

target_minT = pd.read_csv('./date/31088_103.csv')
mix.set_index_date(target_minT, cn.date)
X = pd.read_csv('./date/character_31088.csv')
mix.set_index_date(X, cn.date)

target_train = mix.year_less_eq(target_minT, split_year)
X_train = mix.year_less_eq(X, split_year)

model_train = pf.Model1(X_train, target_train, **param_model)
model_train.save('X_train')

model_test = pf.Model1(X, target_minT, **param_model)
model_test.target = mix.year_great(model_test.target, split_year)
model_test.X_train = mix.year_great(model_test.X_train, split_year)
model_test.save('X_test')