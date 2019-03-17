import pandas as pd
import mix as mix
import db_column_name


# Прогноз температуры на станции с ограничением дат +-N дней
# с усреднением 
# offset can be 9, 21. 33, 45, 57, 69

class Model1:
    def __init__(self, X_train, target, 
                 offset, feature, day_delta):
        self.cn = db_column_name.ColumnName()

        self.columns = [feature]
        self.feature = feature
        self.day_delta = day_delta
        self.offset = offset

        self.target = target

        self.X_train = X_train.drop([self.cn.point], axis=1)
        self.X_train = self.X_train.loc[self.X_train[self.cn.offset] == self.offset, self.columns]

        self.X_train = self.getFeature()

    def getOffset(self, train, date, feature_name):
        offset = mix.group_by_n_day(train, date, self.day_delta)
        offset = offset.add_prefix(feature_name)
        return offset

    def getFeature(self):
        df1 = pd.DataFrame()
        for index, row in self.target.sort_values('actual_date', ascending=False).iterrows():
            #todo df must container only value occurred before
            # f1 = self.getOffset(self.X_train, index, self.feature)
            f2 = self.getOffset(self.target[self.target.index < index], index, 'real ')
            feature = f2 #pd.concat([f1, f2], axis=1)
            df1 = pd.concat([df1, feature], axis=0)
        return df1

    def save(self, name):
        self.X_train[self.cn.value] = self.target[self.cn.value]
        self.X_train.to_csv('./date/' + name)