import pandas as pd
from rectools import Columns
import numpy as np
from datetime import datetime, timedelta
df = pd.read_csv('my_films.csv')
df = df.iloc[:, 2:]
# Преобразуем таблицу из wide в long формат
df_long = df.melt(var_name='title', value_name=Columns.Weight)
print(df_long.head(70))
# Удалим строки без оценок
df_long = df_long.dropna()
print(df.shape)
# Сбросим индексы пользователей
df_long[Columns.User] = df_long.index % df.shape[0]

# Сопоставим каждому фильму уникальный movie_id
unique_titles = df_long['title'].unique()
title_to_id = {title: i for i, title in enumerate(unique_titles)}
df_long[Columns.Item] = df_long['title'].map(title_to_id)

# Если в таблице есть колонка с временными метками — добавим:
# Например, если в оригинальном df был timestamp, можно отдельно сопоставить:
# df_long['timestamp'] = ...

# Создаём timestamp по возрастанию movie_id
date_str = '2016-09-30'

timestamp = pd.to_datetime(date_str)
timestamp2 = pd.to_datetime('2025-05-23')

#print(timestamp2, timestamp, diff)
time1 = timestamp.timestamp()
time2 = timestamp2.timestamp()
diff = time2 - time1
total_movies = len(unique_titles)

movie_id_to_time = {}
for us in range(df.shape[0]):
    for id in range(total_movies):
        #fraction = np.random.random() * ((np.sqrt(id) + np.random.random())/np.sqrt(total_movies))
        fraction = np.random.random()

        movie_id_to_time[(id,us)] = time1 + fraction*diff
        print(movie_id_to_time[(id,us)])
#df_long[Columns.Datetime] = df_long[Columns.Item].map(movie_id_to_time)
df_long[Columns.Datetime] = df_long.apply(
    lambda row: movie_id_to_time.get((row[Columns.Item], row[Columns.User])),
    axis=1)

# Сформируем итоговый датафрейм
df_final = df_long[[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime]]  # добавь 'timestamp', если есть

# Создаём словарь сопоставления movie_id → название
movie_id_map = {i: title for i, title in enumerate(unique_titles)}

# Пример вывода:
print(df_final.head(60))
print(movie_id_map)
df_final.to_csv('long_my_films.csv')
map_df = pd.DataFrame.from_dict(movie_id_map, orient='index')
map_df.to_csv('map_my_films.csv')