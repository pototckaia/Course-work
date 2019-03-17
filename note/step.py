import numpy as np
import pandas as pd

pd.set_option('precision', 10)

df_true = pd.read_csv('./date/31088_103.csv')
df = pd.read_csv('./date/character_31088.csv')
df['actual_date'] = df['actual_date'].astype('datetime64[ns]')

df.sort_values(by="actual_date", ascending=False).head()


# Отбор с использованием моделей
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

x_data_generated, y_data_generated = make_classification()

pipe = make_pipeline(SelectFromModel(estimator=RandomForestClassifier()),
                     LogisticRegression())

lr = LogisticRegression()
rf = RandomForestClassifier()

print(cross_val_score(lr, x_data_generated, y_data_generated, scoring='neg_log_loss').mean())
print(cross_val_score(rf, x_data_generated, y_data_generated, scoring='neg_log_loss').mean())
print(cross_val_score(pipe, x_data_generated, y_data_generated, scoring='neg_log_loss').mean())

-0.184853179322
-0.235652626736
-0.158372952933


 selector = SequentialFeatureSelector(LogisticRegression(), scoring='neg_log_loss', verbose=2, k_features=3, forward=False, n_jobs=-1)

In : selector.fit(x_data_scaled, y_data)

In : selector.fit(x_data_scaled, y_data)




Прогноз минимальной температуры без включения набор предикторов последнего значения наблюденной 
температуры на станции с ограничением дат +- 15 дней с использованием информационного предиктора

Прогноз минимальной температуры с принудительно включенным в набор предикторов 
последнего значения наблюденной температуры на станции с ограничением дат +- 15 дней с использованием информационного предиктора

Прогноз минимальной температуры без принудительно включенного в набор предикторов последнего значения наблюденной температуры 
на станции с ограничением дат +- 15 дней с использованием информационного предиктора
Прогноз минимальной температуры без включения набор предикторов последнего значения наблюденной температуры
 на станции без ограничения дат дней с использованием информационного предиктора
Прогноз минимальной температуры без включения набор предикторов последнего
 значения наблюденной температуры на станции без ограничения дат дней без использования информационного предиктора
Прогноз минимальной температуры без включения набор предикторов последнего значения
 наблюденной температуры на станции без ограничения дат дней без использования информационного предиктора
Прогноз минимальной температуры с принудительно включенным в набор предикторов последним значением наблюденной температуры на станции с ограничением дат +- 15 дней без использования информационного предиктора
Прогноз минимальной температуры с единственным предиктором – смоделированной минимальной температурой с ограничением дат +- 15 дней с использованием информационного предиктора
Прогноз минимальной температуры с единственным предиктором – смоделированной минимальной температурой с ограничением дат +- 15 дней без использования информационного предиктора
Прогноз минимальной температуры с единственным предиктором – смоделированной минимальной температурой без ограничения дат дней с использованием информационного предиктора
Прогноз минимальной температуры с единственным предиктором – смоделированной минимальной температурой без ограничения дат дней без использования информационного предиктора
