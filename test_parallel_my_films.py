from importlib.metadata import metadata

import numpy as np
from itertools import combinations
from copy import deepcopy
from collections import deque
# библиотеки для симуляции и отрисовки выборов
import time
from tqdm import tqdm
import ast
import json
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

def time_split(df, quant = 0.5):
    print("time splitting")
    train_parts = []
    test_parts = []

    for user_id, user_df in df.groupby(Columns.User):
        # Вычисляем квантиль для текущего пользователя
        split_date = user_df[Columns.Datetime].quantile(quant)
        # Разделяем на train и test по split_date
        train_user = user_df[user_df[Columns.Datetime] <= split_date]
        test_user = user_df[user_df[Columns.Datetime] > split_date]
        #print(test_user[Columns.Item].nunique()/user_df[Columns.Item].nunique())
        train_parts.append(train_user)
        test_parts.append(test_user)

    # Объединяем по всем пользователям
    train_df = pd.concat(train_parts)
    test_df = pd.concat(test_parts)
    pivot_df = train_df.pivot_table(index=Columns.User, columns=Columns.Item, values=Columns.Weight)
    print(train_df[Columns.Item].nunique()/df[Columns.Item].nunique())
    return train_df, test_df, pivot_df
def test_GT_light(df_train, df_test, links, pivo, cand_dist, ids_to_num, user_id = 0, size = 10, degrees = 4,
                  weighted = True, rule = 'SNTV', dist_method = 'jaccar', series_rate = 2, metric = True):
    times = {}


    time_0 = time.time()
    recs_test = Recommend_new(links, df_train, pivo, degrees = degrees, remove_rate = 1, series_rate = series_rate,
                              dist_method=dist_method, full_dist=True, ids_to_num=ids_to_num, cand_dist_matrix=cand_dist)
    #times['generation distances_' + weighted*'weighted' + '_' + rule] = time.time() - time_0
    #print('generation distances_' + weighted*'weighted' + '_' + rule + ':', time.time() - time_0)
    time_0 = time.time()
    recos = recs_test.recommendation_voting(user_id, size, rule = rule, weighted = weighted)
    #times['recommendation_' + weighted*'weighted' + '_' + rule] = time.time() - time_0
    #print('recommendation_' + weighted*'weighted' + '_' + rule + ':', time.time() - time_0)
    if metric:
        metrics, weighted_recos = recs_test.metrics(df_test, df_train, user_id)
        return metrics, recos, weighted_recos
    else:
        return recos, times
def full_test_GT_light(combination, user):
    if combination[1] == 'jaccar':
        cand_dist = pd.read_csv('my_films/gened_dists_' + combination[1] + '_' + str(combination[2]) + '.csv', index_col=0).to_numpy()
        ids_to_num_df = pd.read_csv('my_films/gened_dists_ids_' + combination[1] + '_' + str(combination[2]) + '.csv', index_col=0)
        #print(ids_to_num_df)
        ids_to_num = ids_to_num_df[ids_to_num_df.columns[0]].to_dict()
        #print(ids_to_num)
    else:
        cand_dist = pd.read_csv('my_films/gened_dists_' + combination[1] + '.csv', index_col=0).to_numpy()
        ids_to_num_df = pd.read_csv('my_films/gened_dists_ids_' + combination[1] + '.csv', index_col=0)
        ids_to_num = ids_to_num_df[ids_to_num_df.columns[0]].to_dict()
    cur_string = (
                combination[0] + '_' + combination[1] + '_deg=' + str(combination[2]) + '_size=' + str(combination[3]) +
                '_weighted_' * combination[4] + '_antirec_' * (1 - combination[4]) + 'rate=' + str(combination[5]))
    config = dict(zip(params_keys, combination))
    #print(cur_string)
    time_0 = time.time()
    metric, rec, weighted_rec = test_GT_light(df_train, df_test, links_dic, pivo,
                                cand_dist,
                                ids_to_num,
                                user_id=user,
                                metric=True, **config)
    timess = time.time() - time_0
    #print(weighted_rec)
    return metric, weighted_rec, timess, cur_string


rating = pd.read_csv('long_my_films.csv')
movies = pd.read_csv('map_my_films.csv')

links_dic = movies[movies.columns[1]].to_dict()
movies = pd.DataFrame({
    Columns.Item: list(links_dic.keys()),
    'title': list(links_dic.values())
})
times = {}
metrics = {}
recos_dic = {}
all_params_grid = {'rule':['SNTV', 'STV_star', 'STV_basic', 'BnB'],
               'dist_method':['jaccar', 'cosine', 'cosine_hat', 'pearson', 'pearson_hat', 'spearman', 'spearman_hat', 'kendall_hat', 'kendall'],
               'degrees':[4, 2, 3, 5, 6, 7, 8, 9, 10],
               'size':[10, 15, 20, 25, 30],
               'weighted':[True, False],
               'series_rate':[0, 1, 2, 3]}
params_grid = {'rule':['STV_star', 'SNTV'],
               'dist_method':['jaccar', 'cosine', 'kendall'],
               'degrees':[7],
               'size':[10],
               'weighted':[True],
               'series_rate':[0, 1]}
params_keys = params_grid.keys()
params_values = params_grid.values()

df_train, df_test, pivo = time_split(rating, quant=0.75)

#print(links_dic)

# full_test_GT_light(['STV_basic', 'jaccar', 7, 10, False, 0], 0)
# exit()
models_list = ['KNN cosine', 'KNN TF-IDF', 'KNN BM25', 'ALS', 'Random', 'Popular']

recos_ml_dic = {}
recs_test = Recommend(df_train, movies, commit_size=10)
models_dic = {'KNN cosine': recs_test.recs_KNN(commit_size=10, dist_method='cosine'),
              'KNN TF-IDF': recs_test.recs_KNN(commit_size=10, dist_method='TF-IDF'),
              'KNN BM25': recs_test.recs_KNN(commit_size=10, dist_method='BM25'),
              'ALS': recs_test.recs_ALS(commit_size=10),
              'Random': recs_test.recs_Random(commit_size=10),
              'Popular': recs_test.recs_Popular(commit_size=10)}
for mod in models_list:
    models_dic[mod]
    _, recos_ml_dic[mod] = recs_test.metrics(df_test, mod)
for user in rating[Columns.User].unique()[:100]:
    tests = []
    print(user)
    for combination in product(*params_values):
        cur_string = (combination[0] + '_' + combination[1] + '_deg=' + str(combination[2]) + '_size=' + str(
            combination[3]) +
                      '_weighted_' * combination[4] + '_antirec_' * (1 - combination[4]) + 'rate=' + str(
                    combination[5]))
        times[cur_string] = []
        tests.append(combination)

    results = Parallel(n_jobs=-1, verbose=1)(
        delayed(full_test_GT_light)(combination, user)
        for combination in tests
    )
    recos_list = []
    recos_keys_list = []

    for metric, rec, timess, cur_string in results:
        metrics[cur_string] = metric
        recos_dic[cur_string] = rec
        recos_list.append(rec)
        recos_keys_list.append(cur_string)
        times[cur_string].append(timess)


    for mod in models_list:
        rec_ml_df = recos_ml_dic[mod]
        rec_ml_df = rec_ml_df[rec_ml_df[Columns.User] == user]
        rec_ml_df = rec_ml_df.reset_index(drop=True)
        rec_ml_df = rec_ml_df.drop(Columns.User, axis=1)
        recos_list.append(rec_ml_df)
        recos_keys_list.append(mod)


    user_recos = pd.concat(recos_list, axis=1, keys=recos_keys_list)
    #metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    #metrics_df = metrics_df.T
    #recos_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in recos_dic.items()]))
    #metrics_df.to_csv('my_films/test4/metrics_user' + str(user) + '.csv', index=True)
    user_recos.to_csv('my_films/test4/recos_' + str(user) + '.csv')


# for key, item in times.items():
#     #print(key, np.array(item).mean())
#     times[key] = np.array(item).mean()
# times_df = pd.DataFrame.from_dict(times, orient="index")
# times_df.to_csv('my_films/test4/times_mac.csv')