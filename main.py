# библиотеки для BnB алгоритма поиска оптимальных кандидатов через pseudo bollean polynomes
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

def time_split(raiting, user_id = 1, quant = 0.5):
    print("time splitting")
    df = raiting.rename(columns={'userId': Columns.User, 'movieId': Columns.Item, 'rating': Columns.Weight, 'timestamp': Columns.Datetime})
    #print(df.head(10))
    split_date = df[df[Columns.User] == user_id][Columns.Datetime].quantile(quant)
    #print('split date', split_date)

    train = df[df[Columns.Datetime] <= split_date]
    test = df[df[Columns.Datetime] > split_date]
    #print('train:' ,train[train[Columns.User] == user_id])
    #print('test:', test[test[Columns.User] == user_id])
    pivot_df = train.pivot_table(index=Columns.User, columns=Columns.Item, values=Columns.Weight)
    return train, test, pivot_df

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
def full_test_GT(combination):
    cur_string = (
                combination[0] + '_' + combination[1] + '_deg=' + str(combination[2]) + '_size=' + str(combination[3]) +
                '_weighted_' * combination[4] + '_antirec_' * (1 - combination[4]) + 'rate=' + str(combination[5]))
    config = dict(zip(params_keys, combination))
    print(cur_string)
    time_0 = time.time()
    metric, rec = test_GT(df_train, df_test, links_dic, pivo, user_id=user, metric=True, **config)
    timess = time.time() - time_0
    return metric, rec, timess, cur_string
def test_ML(ratings, ratings_test, movies, user_id = 0, metric = True):
    metrics_values = {}
    recos = {}
    times = {}
    print(ratings.shape)
    #print(ratings.head(10))
    time_0 = time.time()
    recs_test = Recommend(ratings, movies)
    recos['KNN'] = recs_test.recs_KNN(user_id=user_id, commit_size=10)
    times['KNN'] = time.time() - time_0
    print("KNN:", time.time() - time_0)
    time_0 = time.time()
    recos['Random'] = recs_test.recs_Random(user_id=user_id, commit_size=10)
    times['Random'] = time.time() - time_0
    print("Random:", time.time() - time_0)
    time_0 = time.time()
    recos['Popular'] = recs_test.recs_Popular(user_id=user_id, commit_size=10)
    times['Popular'] = time.time() - time_0
    print("Popular:", time.time() - time_0)
    time_0 = time.time()
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
        return all_metrics, recos, times
    else:
        return recos, times

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

# C = np.array([[7, 15, 10, 7, 10], [10, 17, 9, 11, 22], [16, 7, 6, 18, 14], [11, 7, 6, 12, 8]])
# Votes = np.argsort(C, axis=0)
# print(C)
# print(Votes)
# recs = Reccomend(Vote_matrix=Votes, degrees=2)
# recs.App_Sets()
# recs.Candidates_dists()

#rating = pd.read_csv('rate3.csv')
rating = pd.read_csv('archive/ratings_small.csv')
#print(rating)

movies = pd.read_csv('archive/links_small.csv')
metadata = pd.read_csv('archive/movies_metadata.csv', low_memory=False)
#print(metadata.head(10))
#print(movies.head(10))

movies['original_title'] = metadata['original_title'].reindex(movies.index, fill_value='unknown')
links_dic = dict(zip(movies['movieId'], movies['original_title']))
#print(links_dic)
#headers = list(rating.columns)[2:]
#print('headers', headers)
#movies_list = []
#for item_id, item in enumerate(headers):
#    movies_list.append([item_id, item])
#movies = pd.DataFrame(movies_list, columns=[Columns.Item, "title"])
#print(movies)
user = 3
df_train, df_test, pivo = time_split(rating, user_id=user, quant=0.75)
#for id in df_test[Columns.Item]:
#    print('fim', links_dic[id])
#print(df_train)
#print(df_test)
#print(df)
all_metrics, recos = test_GT(df_train, df_test, links_dic, pivo, degrees = 2, user_id = user, size = 10,
                             weighted=True, rule = 'STV_star', dist_method='jaccar', metric = True)
print(all_metrics)
print(recos)
print(time)

exit()
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
               'dist_method':['jaccar', 'pearson', 'spearman', 'cosine', 'kendall'],
               'degrees':[2, 3, 4, 5, 6, 7, 8],
               'size':[10, 20],
               'weighted':[True, False],
               'series_rate':[0, 2]}
params_keys = params_grid.keys()
params_values = params_grid.values()
step = 1
tests = []
for user in rating['userId'].unique()[1:51]:
    df_train, df_test, pivo = time_split(rating, user_id=user, quant=0.75)
    time_0 = time.time()
    print(step)
    for combination in product(*params_values):
        cur_string = (combination[0] + '_' + combination[1] + '_deg=' + str(combination[2]) + '_size=' + str(
            combination[3]) +
                      '_weighted_' * combination[4] + '_antirec_' * (1 - combination[4]) + 'rate=' + str(
                    combination[5]))
        times[cur_string] = []
        tests.append(combination)

    results = Parallel(n_jobs=-1)(
        delayed(full_test_GT)(combination)
        for combination in tests
    )
    for metric, rec, timess, cur_string in results:
        metrics[cur_string] = metric
        recos_dic[cur_string] = rec
        times[cur_string].append(timess)
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    metrics_df = metrics_df.T
    recos_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in recos_dic.items()]))
    # recos_df = pd.DataFrame.from_dict(recos_dic)
    print(recos_df)
    print(metrics_df)
    metrics_df.to_csv('GT/STV_metrics_user' + str(user) + '.csv', index=True)
    recos_df.to_csv('GT/STV_recos_' + str(user) + '.csv')
    step += 1

for user in rating['userId'].unique()[1:51]:
    df_train, df_test, pivo = time_split(rating, user_id=user, quant=0.75)
    time_0 = time.time()
    for combination in product(*params_values):
        #cur_string = (rule + '_' + dist_method + '_deg=' + str(degrees) + '_size=' + str(size) + weighted*'_weighted_' + (1-weighted)*'_antirec_' + series*('series' + str(series_rate)) + (1-series)*'single')
        cur_string = (combination[0] + '_' + combination[1] + '_deg=' + str(combination[2]) + '_size=' + str(combination[3]) +
                      '_weighted_'*combination[4] + '_antirec_'*(1 - combination[4]) + 'rate=' + str(combination[5]))
        times[cur_string] = []
        config = dict(zip(params_keys, combination))
        print(str(step) + '/' + str(30), cur_string)
        time_0 = time.time()
        (metrics[cur_string],
         recos_dic[cur_string]) = test_GT(df_train, df_test, links_dic, pivo,  user_id=user, metric=True, **config)
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

user = 0

ratings_ML, ratings_test_ML, ratings_GT = zero_split(rating)
metrics, recos_dic, full_times = test_ML(ratings_ML, ratings_test_ML, movies, user)
for rule in ['SNTV', 'BnB']:
    for key in ['series', 'remove_bad']:
        metrics['Election_' + key + '_' + rule], recos_dic['Election_' + key + '_' + rule], temp_dic = test_my(ratings_GT,
                                                                                                     ratings_test_ML,
                                                                                                     ratings_ML, user,
                                                                                                     rule=rule,
                                                                                                     method=key)
        full_times.update(temp_dic)
full_times_df = pd.DataFrame.from_dict(full_times, orient='index', columns=['value'])
full_times_df.to_csv('full_times.csv')
#metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
#metrics_df = metrics_df.T
#recos_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in recos_dic.items()]))

#print(recos_df)
#print(metrics_df)

#metrics_df.to_csv('metrics_zero_cut.csv', index=True)
#recos_df.to_csv('recos_zero_cut_' + str(user)  + '.csv')

#user = 39
#real_recos(rating, user)
print('finish')
#print(testing_BnB.Scores)
'''