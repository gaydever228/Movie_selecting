from importlib.metadata import metadata

import numpy as np
from itertools import combinations
from copy import deepcopy
from collections import deque
# библиотеки для симуляции и отрисовки выборов
import time
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
        print(test_user[Columns.Item].nunique()/user_df[Columns.Item].nunique())
        train_parts.append(train_user)
        test_parts.append(test_user)

    # Объединяем по всем пользователям
    train_df = pd.concat(train_parts)
    test_df = pd.concat(test_parts)
    pivot_df = train_df.pivot_table(index=Columns.User, columns=Columns.Item, values=Columns.Weight)
    print(train_df[Columns.Item].nunique()/df[Columns.Item].nunique())
    return train_df, test_df, pivot_df

def gen_dist(dist_method):

    times = {}
    flag = 0
    for degrees in params_grid['degrees']:
        print('comp dist', dist_method, degrees)
        if dist_method == 'jaccar' or flag == 0:
            flag = 1
            time0 = time.time()
            dist_gen = Recommend_new(links_dic, df_train, pivo, degrees=degrees, remove_rate=1,
                                     dist_method=dist_method, full_dist=True)
            times[degrees] = time.time() - time0
            cand, ids = dist_gen.distances()
            print('comp dist', dist_method, degrees, 'export finished')
            cand_dist_df = pd.DataFrame(cand)
            ids_to_num_df = pd.DataFrame.from_dict(ids, orient='index')
            if dist_method == 'jaccar':
                #safe_to_csv(cand_dist_df, 'GT/gened_dists_' + dist_method + '_' + str(degrees) + '.csv')
                cand_dist_df.to_csv('my_films/gened_dists_' + dist_method + '_' + str(degrees) + '.csv')
                ids_to_num_df.to_csv('my_films/gened_dists_ids_' + dist_method + '_' + str(degrees) + '.csv')
                #safe_to_csv(ids_to_num_df, 'GT/gened_dists_ids_' + dist_method + '_' + str(degrees) + '.csv')
            else:
                cand_dist_df.to_csv('my_films/gened_dists_' + dist_method + '.csv')
                ids_to_num_df.to_csv('my_films/gened_dists_ids_' + dist_method + '.csv')

        # else:
        #     cand_dist[degrees], ids_to_num[degrees] = (
        #         cand_dist[params_grid['degrees'][0]],
        #         ids_to_num[params_grid['degrees'][0]])

    return dist_method, times
def safe_to_csv(df, path):
    def serialize(val):
        if isinstance(val, (list, dict)):
            return json.dumps(val)
        return val

    df_serialized = df.map(serialize)
    df_serialized.to_csv(path)
def safe_from_csv(path):
    def parse_obj(obj):
        if isinstance(obj, dict):
            try:
                return {int(k): v for k, v in obj.items()}
            except ValueError:
                return obj
        return obj
    def deserialize(val):
        try:
            obj = json.loads(val)
            return parse_obj(obj)
        except (json.JSONDecodeError, TypeError):
            try:
                return int(val)
            except (ValueError, TypeError):
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return val

    df = pd.read_csv(path, index_col=0)
    return df.map(deserialize)
# rating = pd.read_csv('archive/ratings_small.csv')
# #print(rating)
# #rating['movieId'] = rating['movieId'].astype(int)
# #rating['userId'] = rating['userId'].astype(int)
# print('до', rating['movieId'].nunique(), rating['userId'].nunique())
# item_user_counts = rating.groupby('movieId')['userId'].nunique()
# valid_items = item_user_counts[item_user_counts > 2].index
# rating = rating[rating['movieId'].isin(valid_items)]
#
# user_item_counts = rating.groupby('userId')['movieId'].nunique()
# valid_users = user_item_counts[user_item_counts > 2].index
# rating = rating[rating['userId'].isin(valid_users)]
# print('после', rating['movieId'].nunique(), rating['userId'].nunique())
# movies = pd.read_csv('archive/links_small.csv')
# #movies['movieId'] = movies['movieId'].astype(int)
# metadata = pd.read_csv('archive/movies_metadata.csv', low_memory=False)
# #print(metadata.head(10))
# #print(movies.head(10))
#
# movies['original_title'] = metadata['original_title'].reindex(movies.index, fill_value='unknown')
# links_dic = dict(zip(movies['movieId'], movies['original_title']))
rating = pd.read_csv('long_my_films.csv')
movies = pd.read_csv('map_my_films.csv')

links_dic = movies[movies.columns[1]].to_dict()
times = {}
metrics = {}
recos_dic = {}
all_params_grid = {'rule':['SNTV', 'STV_star', 'STV_basic', 'BnB'],
               'dist_method':['jaccar', 'cosine', 'cosine_hat', 'pearson', 'pearson_hat', 'spearman', 'spearman_hat', 'kendall_hat', 'kendall'],
               'degrees':[2, 3, 4, 5, 6, 7, 8, 9, 10],
               'size':[10, 15, 20, 25, 30],
               'weighted':[True, False],
               'series_rate':[0, 1, 2, 3]}
params_grid = {'rule':['SNTV', 'STV_basic', 'STV_star', 'BnB'],
               'dist_method':['jaccar', 'cosine', 'cosine_hat', 'pearson', 'pearson_hat', 'spearman', 'spearman_hat', 'kendall_hat', 'kendall'],
               'degrees':[2, 3, 4, 5, 6, 7, 8, 9, 10],
               'size':[5, 10, 15, 20, 25, 30],
               'weighted':[False, True],
               'series_rate':[0, 1, 2, 3]}
params_keys = params_grid.keys()
params_values = params_grid.values()

df_train, df_test, pivo = time_split(rating, quant=0.75)



cand_dist = {}
ids_to_num = {}
time_dist = {}
results = Parallel(n_jobs=-1)(
    delayed(gen_dist)(dist_method)
    for dist_method in params_grid['dist_method']
)

# for dist_method, t in results:
#     time_dist[dist_method] = t[params_grid['degrees'][0]]
#cand_dist_df = pd.DataFrame.from_dict(cand_dist)
#ids_to_num_df = pd.DataFrame.from_dict(ids_to_num)
#cand_dist_df.to_csv('GT/gened_dists2.csv')
#ids_to_num_df.to_csv('GT/gened_dists_ids2.csv')
#safe_to_csv(cand_dist_df, 'GT/gened_dists.csv')
#safe_to_csv(ids_to_num_df, 'GT/gened_dists_ids.csv')

# times_dist_df = pd.DataFrame.from_dict(time_dist, orient='index')
# times_dist_df.to_csv('GT/gen_dist_times.csv')
