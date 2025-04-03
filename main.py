# библиотеки для BnB алгоритма поиска оптимальных кандидатов через pseudo bollean polynomes
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

from ml_recs import Recommend

rng = Generator(PCG64())
import copy
from tqdm import tqdm
from rich.progress import Progress
import csv
from scipy.optimize import curve_fit
from election import election
from PBF import PBF, BnB, bound, branch
from Test import Test
from recsys import Reccomend
from rectools import Columns
from datetime import datetime, timedelta
# C = np.array([[7, 15, 10, 7, 10], [10, 17, 9, 11, 22], [16, 7, 6, 18, 14], [11, 7, 6, 12, 8]])
# Votes = np.argsort(C, axis=0)
# print(C)
# print(Votes)
# recs = Reccomend(Vote_matrix=Votes, degrees=2)
# recs.App_Sets()
# recs.Candidates_dists()
rating = pd.read_csv('rate3.csv')
headers = list(rating.columns)[2:]
#print('headers', headers)
movies_list = []
for item_id, item in enumerate(headers):
    movies_list.append([item_id, item])
movies = pd.DataFrame(movies_list, columns=[Columns.Item, "title"])
#print(movies)
def random_split(rating):
    ratings_list = []
    ratings_test_list = []
    print("random splitting")
    user_id = 0
    rating_cut = copy.deepcopy(rating)
    for user, rates in rating.iterrows():
        # print(user, rates[1])

        time = rates[0]
        date_str_clean = time.replace("GMT+3", "").strip()
        dt = datetime.strptime(date_str_clean, "%Y/%m/%d %I:%M:%S %p")
        dt64 = np.datetime64(dt)
        item_id = 0
        for item in rates[2:]:
            if not pd.isna(item):
                cut = np.random.binomial(1, 0.2)
                #print(user_id, item_id, item)
                #print(rating_cut.iat[user, item_id + 2])
                if cut == 0:
                    ratings_list.append([user_id, item_id, item, dt64])
                else:
                    ratings_test_list.append([user_id, item_id, item, dt64])
                    rating_cut.iat[user, item_id + 2] = np.nan

            item_id += 1
        user_id += 1
    ratings = pd.DataFrame(ratings_list, columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime])
    ratings_test = pd.DataFrame(ratings_test_list,
                                columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime])
    return ratings, ratings_test, rating_cut
def zero_split(rating, limit = 120):
    ratings_list = []
    ratings_test_list = []
    print("zero splitting")
    user_id = 0
    rating_cut = copy.deepcopy(rating)
    for user, rates in rating.iterrows():
        # print(user, rates[1])
        time = rates[0]
        date_str_clean = time.replace("GMT+3", "").strip()
        dt = datetime.strptime(date_str_clean, "%Y/%m/%d %I:%M:%S %p")
        dt64 = np.datetime64(dt)
        item_id = 0
        for item in rates[2:]:
            if not pd.isna(item):
                if user_id != 0 or item_id < limit:
                    ratings_list.append([user_id, item_id, item, dt64])
                else:
                    ratings_test_list.append([user_id, item_id, item, dt64])
                    rating_cut.iat[user, item_id + 2] = np.nan

            item_id += 1
        user_id += 1
    ratings = pd.DataFrame(ratings_list, columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime])
    ratings_test = pd.DataFrame(ratings_test_list,
                                columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime])
    return ratings, ratings_test, rating_cut
def test_my(df, df_test, df_train, user_id = 0, method = 'series', rule = 'SNTV', metric = True):

    #heh = np.array(headers)
    #mask = np.char.endswith(heh, ".1")
    #result = heh[mask]

    #print("дубликаты:", result)
    df = df.T
    #print(df.head(10))

    raiting1 = np.array(df.values[2:], dtype = float)
    #print(raiting1)
    #print(type(raiting1[0, 0]), type(raiting1[1, 1]))


    #for c in raiting1[1]:
    #    print(c)
    #    print(np.isnan(c))
    #    print(c>10)
    recs_test = Reccomend(headers, len(raiting1[0]), len(raiting1), degrees=3, raiting=raiting1)
    recos = recs_test.recommendation_voting(user_id, 10, rule = rule, method=method)
    if metric:
        metrics = recs_test.metrics(df_test, df_train, user_id)
        return metrics, recos
    else:
        return recos

def test_ML(ratings, ratings_test, movies, user_id = 0, metric = True):
    metrics_values = {}
    recos = {}
    print(ratings.shape)
    #print(ratings.head(10))
    recs_test = Recommend(ratings, movies)
    recos['KNN'] = recs_test.recs_KNN(user_id=user_id, commit_size=10)
    recos['Random'] = recs_test.recs_Random(user_id=user_id, commit_size=10)
    recos['Popular'] = recs_test.recs_Popular(user_id=user_id, commit_size=10)
    if metric:
        metrics_values['KNN'] = recs_test.metrics(ratings_test, 'KNN')
        metrics_values['Random'] = recs_test.metrics(ratings_test, 'Random')
        metrics_values['Popular'] = recs_test.metrics(ratings_test, 'Popular')

        all_metrics = {}
        for key, item in metrics_values.items():
            all_metrics[key] = {}
            for key2, item2 in item.items():
                print('item2', item2)
                all_metrics[key][key2] = item2[user_id]
        return all_metrics, recos
    else:
        return recos

def real_recos(rating, user = 0):
    user_name = str(rating.iat[user, 1])
    print(user_name)
    ratings_ML, ratings_test_ML, ratings_GT = zero_split(rating, limit=200)
    recos_dic = test_ML(ratings_ML, ratings_test_ML, movies, user, metric=False)
    for key in ['series', 'remove_bad']:
        recos_dic['Election_' + key] = test_my(ratings_GT, ratings_test_ML, ratings_ML, user, method=key, metric=False)

    recos_df = pd.DataFrame.from_dict(recos_dic)
    #print(recos_df)
    recos_df.to_csv('recos_' + user_name + str(user) + '.csv')
#recs_test.App_Sets_from_raiting3(raiting1)
#recs_gen.App_Sets()
#recs_test.Candidates_dists()
#print(recs_gen.real_cand_dist)
#testing_score = election(30, 10, 5,  gen = True)
#testing_BnB.STV_test()
#testing_BnB.BnB_level_V()
#testing_score.STV_rule()
#testing_score.SNTV_rule()
#testing_score.BnB_rule(tol =  0.5, level = 2)

'''
user = 0
times = {'ML': [], 'series_SNTV': [], 'remove_bad_SNTV': [], 'series_BnB': [], 'remove_bad_BnB': []}
for i in range(33, 40):
    ratings_ML, ratings_test_ML, ratings_GT = random_split(rating)
    time_0 = time.time()
    metrics, recos_dic = test_ML(ratings_ML, ratings_test_ML, movies, user)
    times['ML'].append(time.time() - time_0)
    for rule in ['SNTV', 'BnB']:
        for key in ['series', 'remove_bad']:
            time_0 = time.time()
            metrics['Election_' + key + '_' + rule], recos_dic['Election_' + key + '_' + rule] = test_my(ratings_GT, ratings_test_ML, ratings_ML, user, rule = rule, method = key)
            times[key + '_' + rule].append(time.time() - time_0)
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    metrics_df = metrics_df.T
    recos_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in recos_dic.items()]))
    #recos_df = pd.DataFrame.from_dict(recos_dic)
    print(recos_df)
    print(metrics_df)
    metrics_df.to_csv('metrics_random_cut_' + str(i) + '.csv', index=True)
    recos_df.to_csv('recos_' + str(user)  + '_' + str(i) + '.csv')

for key, item in times.items():
    print(key, np.array(item).mean())
times_df = pd.DataFrame.from_dict(times)
times_df.to_csv('times.csv')
'''
user = 0
ratings_ML, ratings_test_ML, ratings_GT = zero_split(rating)
metrics, recos_dic = test_ML(ratings_ML, ratings_test_ML, movies, user)
for rule in ['SNTV', 'BnB']:
    for key in ['series', 'remove_bad']:
        metrics['Election_' + key + '_' + rule], recos_dic['Election_' + key + '_' + rule] = test_my(ratings_GT,
                                                                                                     ratings_test_ML,
                                                                                                     ratings_ML, user,
                                                                                                     rule=rule,
                                                                                                     method=key)
metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
metrics_df = metrics_df.T
recos_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in recos_dic.items()]))
#recos_df = pd.DataFrame.from_dict(recos_dic)
print(recos_df)
print(metrics_df)
metrics_df.to_csv('metrics_zero_cut.csv', index=True)
recos_df.to_csv('recos_zero_cut_' + str(user)  + '.csv')

#user = 39
#real_recos(rating, user)
print('finish')
#print(testing_BnB.Scores)