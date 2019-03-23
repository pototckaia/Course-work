import pandas as pd
import mix as mix
import db_column_name


# Прогноз температуры на станции
# offset can be 9, 21. 33, 45, 57, 69

class Model1:
    def __init__(self, X_train, target, 
                 offset):
        self.cn = db_column_name.ColumnName()

        self.offset = offset

        self.target = target

        self.X_train = X_train.drop([self.cn.point], axis=1)
        self.X_train = self.X_train.loc[self.X_train[self.cn.offset] == self.offset]

    def save(self, name):
        self.X_train[self.cn.value] = self.target[self.cn.value]
        self.X_train.to_csv('./date/' + name + '.csv')