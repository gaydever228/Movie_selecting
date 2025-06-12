from importlib.metadata import metadata

import numpy as np
from itertools import combinations
from copy import deepcopy
from collections import deque
# библиотеки для симуляции и отрисовки выборов
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from numpy.random import Generator, PCG64
import pandas as pd
pd.set_option('display.max_columns', 100)
import math
from random import sample
import random
from itertools import product
from ml_recs import Recommend
from joblib import Parallel, delayed
rng = Generator(PCG64())
import copy
from tqdm import tqdm
from rich.progress import Progress
import csv
from scipy.optimize import curve_fit
from election import election
from PBF import PBF, BnB, bound, branch
from Test import Test
from recsys import Recommend_new
from rectools import Columns
from datetime import datetime, timedelta

def time_split(raiting, quant = 0.5):
    print("time splitting")
    train_parts = []
    test_parts = []
    df = raiting.rename(columns={'userId': Columns.User, 'movieId': Columns.Item, 'rating': Columns.Weight,
                                 'timestamp': Columns.Datetime})
    for user_id, user_df in df.groupby(Columns.User):
        # Вычисляем квантиль для текущего пользователя
        split_date = user_df[Columns.Datetime].quantile(quant)
        # Разделяем на train и test по split_date
        train_user = user_df[user_df[Columns.Datetime] <= split_date]
        test_user = user_df[user_df[Columns.Datetime] > split_date]
        train_parts.append(train_user)
        test_parts.append(test_user)

    # Объединяем по всем пользователям
    train_df = pd.concat(train_parts)
    test_df = pd.concat(test_parts)
    pivot_df = train_df.pivot_table(index=Columns.User, columns=Columns.Item, values=Columns.Weight)
    return train_df, test_df, pivot_df
def test_GT(df_train, df_test, links, pivo, user_id = 0, size = 10, degrees = 4, series = True, weighted = True, rule = 'SNTV', dist_method = 'jaccar', series_rate = 2, metric = True):
    times = {}
    time_0 = time.time()
    recs_test = Recommend_new(links, df_train, pivo, degrees = degrees, remove_rate = 1, series_rate = series_rate, dist_method=dist_method)
    #times['generation distances_' + weighted*'weighted' + '_' + rule] = time.time() - time_0
    print('generation distances_' + weighted*'weighted' + '_' + rule + ':', time.time() - time_0)
    time_0 = time.time()
    recos = recs_test.recommendation_voting(user_id, size, rule = rule, weighted = weighted)
    #times['recommendation_' + weighted*'weighted' + '_' + rule] = time.time() - time_0
    print('recommendation_' + weighted*'weighted' + '_' + rule + ':', time.time() - time_0)
    if metric:
        metrics = recs_test.metrics(df_test, df_train, user_id)
        return metrics, recos
    else:
        return recos, times
def test_GT_light(df_train, df_test, links, pivo, cand_dist, ids_to_num, user_id = 0, size = 10, degrees = 4, series = True, weighted = True, rule = 'SNTV', dist_method = 'jaccar', series_rate = 2, metric = True):
    times = {}
    time_0 = time.time()
    recs_test = Recommend_new(links, df_train, pivo, degrees = degrees, remove_rate = 1, series_rate = series_rate,
                              dist_method=dist_method, full_dist=True, ids_to_num=ids_to_num, cand_dist_matrix=cand_dist)
    #times['generation distances_' + weighted*'weighted' + '_' + rule] = time.time() - time_0
    print('generation distances_' + weighted*'weighted' + '_' + rule + ':', time.time() - time_0)
    time_0 = time.time()
    recos = recs_test.recommendation_voting(user_id, size, rule = rule, weighted = weighted)
    #times['recommendation_' + weighted*'weighted' + '_' + rule] = time.time() - time_0
    print('recommendation_' + weighted*'weighted' + '_' + rule + ':', time.time() - time_0)
    if metric:
        metrics = recs_test.metrics(df_test, df_train, user_id)
        return metrics, recos
    else:
        return recos, times
rating = pd.read_csv('archive/ratings_small.csv')
#print(rating)
print('до', rating['movieId'].nunique(), rating['userId'].nunique())
item_user_counts = rating.groupby('movieId')['userId'].nunique()
valid_items = item_user_counts[item_user_counts > 2].index
rating = rating[rating['movieId'].isin(valid_items)]

user_item_counts = rating.groupby('userId')['movieId'].nunique()
valid_users = user_item_counts[user_item_counts > 2].index
rating = rating[rating['userId'].isin(valid_users)]
print('после', rating['movieId'].nunique(), rating['userId'].nunique())
movies = pd.read_csv('archive/links_small.csv')
metadata = pd.read_csv('archive/movies_metadata.csv', low_memory=False)
#print(metadata.head(10))
#print(movies.head(10))

movies['original_title'] = metadata['original_title'].reindex(movies.index, fill_value='unknown')
links_dic = dict(zip(movies['movieId'], movies['original_title']))

times = {}
metrics = {}
recos_dic = {}
all_params_grid = {'rule':['SNTV', 'STV_star', 'STV_basic', 'BnB'],
               'dist_method':['jaccar', 'cosine', 'cosine_hat', 'pearson', 'pearson_hat', 'spearman', 'spearman_hat', 'kendall', 'kendall_hat'],
               'degrees':[2, 3, 4, 5, 6, 7, 8],
               'size':[10, 15, 20, 25, 30],
               'weighted':[True, False],
               'series_rate':[0, 1, 2, 3]}
params_grid = {'rule':['STV_star', 'STV_basic'],
               'dist_method':['jaccar', 'spearman', 'pearson', 'cosine', 'kendall'],
               'degrees':[2, 3, 4, 5],
               'size':[10, 20],
               'weighted':[True, False],
               'series_rate':[0, 2]}
params_keys = params_grid.keys()
params_values = params_grid.values()
step = 1
df_train, df_test, pivo = time_split(rating, quant=0.75)
cand_dist = {}
ids_to_num = {}
for dist_method in params_grid['dist_method']:
    cand_dist[dist_method] = {}
    ids_to_num[dist_method] = {}

    for degrees in params_grid['degrees']:
        print(degrees, dist_method)
        if dist_method == 'jaccar' or len(cand_dist[dist_method]) == 0:
            dist_gen = Recommend_new(links_dic, df_train, pivo, degrees=degrees, remove_rate=1,
                                  dist_method=dist_method, full_dist=True)

            cand_dist[dist_method][degrees], ids_to_num[dist_method][degrees] = dist_gen.distances()
        else:
            cand_dist[dist_method][degrees], ids_to_num[dist_method][degrees] = (
                cand_dist[dist_method][params_grid['degrees'][0]],
                ids_to_num[dist_method][params_grid['degrees'][0]])
for user in rating['userId'].unique()[2:20]:

    for combination in product(*params_values):
        cur_string = (combination[0] + '_' + combination[1] + '_deg=' + str(combination[2]) + '_size=' + str(
            combination[3]) +
                      '_weighted_' * combination[4] + '_antirec_' * (1 - combination[4]) + 'rate=' + str(
                    combination[5]))
        times[cur_string] = []
    for combination in product(*params_values):
        #cur_string = (rule + '_' + dist_method + '_deg=' + str(degrees) + '_size=' + str(size) + weighted*'_weighted_' + (1-weighted)*'_antirec_' + series*('series' + str(series_rate)) + (1-series)*'single')
        cur_string = (combination[0] + '_' + combination[1] + '_deg=' + str(combination[2]) + '_size=' + str(combination[3]) +
                      '_weighted_'*combination[4] + '_antirec_'*(1 - combination[4]) + 'rate=' + str(combination[5]))

        config = dict(zip(params_keys, combination))
        print(str(step) + '/' + str(30), cur_string)
        time_0 = time.time()
        (metrics[cur_string],
         recos_dic[cur_string]) = test_GT_light(df_train, df_test, links_dic, pivo,
                                                cand_dist[combination[1]][combination[2]],
                                                ids_to_num[[combination[1]][combination[2]]],
                                                user_id=user, metric=True, **config)
        times[cur_string].append(time.time() - time_0)
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    metrics_df = metrics_df.T
    recos_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in recos_dic.items()]))
    #recos_df = pd.DataFrame.from_dict(recos_dic)
    print(recos_df)
    print(metrics_df)
    metrics_df.to_csv('GT/STV_metrics_user' + str(user) + '.csv', index=True)
    recos_df.to_csv('GT/STV_recos_' + str(user) + '.csv')
    step += 1

#for key, item in times.items():
#    print(key, np.array(item).mean())
times_df = pd.DataFrame.from_dict(times)
times_df.to_csv('GT/STV_times.csv')