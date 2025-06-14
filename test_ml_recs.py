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

def test_ML(ratings, ratings_test, titles, k = 10, metric = True):

    metrics_values = {}
    models_list = ['KNN cosine', 'KNN TF-IDF', 'KNN BM25', 'ALS', 'Random', 'Popular']

    recos = {}
    times = {}
    print('shape', ratings.shape)
    #print(ratings.head(10))
    time_0 = time.time()
    recs_test = Recommend(ratings, titles, commit_size=k)
    models_dic = {'KNN cosine': recs_test.recs_KNN(commit_size=k, dist_method='cosine'),
                  'KNN TF-IDF': recs_test.recs_KNN(commit_size=k, dist_method='TF-IDF'),
                  'KNN BM25': recs_test.recs_KNN(commit_size=k, dist_method='BM25'),
                  'ALS': recs_test.recs_ALS(commit_size=k),
                  'Random': recs_test.recs_Random(commit_size=k),
                  'Popular': recs_test.recs_Popular(commit_size=k)}
    for mod in models_list:
        time_0 = time.time()
        models_dic[mod]

        for user in tqdm(ratings[Columns.User].unique()):
            recos[mod] = recs_test.user_recs(user, mod)
            metrics_values[mod] = recs_test.metrics(ratings_test, mod)
        times[mod] = time.time() - time_0
        print('meh1', mod, times[mod])
    for user in tqdm(ratings[Columns.User].unique()):
        all_metrics = {}
        for key, item in metrics_values.items():
            all_metrics[key] = {}
            for key2, item2 in item.items():
                #print('item2', item2)
                all_metrics[key][key2] = item2[user]
        metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index')
        metrics_df = metrics_df.T
        recos_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in recos.items()]))
        metrics_df.to_csv('my_films/test_ML/metrics_user' + str(user) + '.csv', index=True)
        recos_df.to_csv('my_films/test_ML/recos_' + str(user) + '.csv')
    return times



rating = pd.read_csv('long_my_films.csv')
movies = pd.read_csv('map_my_films.csv')
movies.columns = [Columns.Item, "title"]
#links_dic = movies[movies.columns[1]].to_dict()
df_train, df_test, pivo = time_split(rating, quant=0.75)



times = test_ML(df_train, df_test, movies, 10)

times_df = pd.DataFrame.from_dict(times, orient="index")
times_df.to_csv('my_films/test_ML/times.csv')