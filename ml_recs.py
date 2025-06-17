from os import environ

from implicit.nearest_neighbours import TFIDFRecommender, ItemItemRecommender, BM25Recommender, CosineRecommender

from pandas.core.interchange.dataframe_protocol import ColumnNullType


environ["OPENBLAS_NUM_THREADS"] = "1"
from implicit.als import AlternatingLeastSquares

import warnings

import time

from pathlib import Path
from pprint import pprint

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from rectools import Columns
from rectools.dataset import Dataset
from rectools.models import (LightFMWrapperModel,
                             ImplicitALSWrapperModel, DSSMModel, PopularModel, RandomModel,
                             PopularInCategoryModel, PopularInCategoryModel, PureSVDModel, ImplicitItemKNNWrapperModel)

#from lightfm import LightFM
from rectools.metrics import Precision, Recall, MAP, calc_metrics, MeanInvUserFreq, Serendipity, NDCG
sns.set_theme(style="whitegrid")

class Recommend():
    def __init__(self, rating, movies, commit_size = 10):
        self.dataset = Dataset.construct(rating)
        self.rating = rating
        self.movies = movies
        self.models = {}
        self.recos = {}
        self.k = commit_size
        #model1 = ImplicitALSWrapperModel()

    def user_recs(self, user_id, model_name):
        user_viewed = self.rating.query("user_id == @user_id").merge(self.movies, on="item_id")

        user_recos = self.recos[model_name].query("user_id == @user_id").merge(self.movies, on="item_id")

        return list(user_recos['title'])
    def recs_KNN(self, commit_size = 10, dist_method = 'TF-IDF'):

        if dist_method == 'TF-IDF':
            recomender = itm_tfidf = TFIDFRecommender(K=commit_size)
        elif dist_method == 'BM25':
            recomender = BM25Recommender(K=commit_size, K1=1.2, B=0.75)
        elif dist_method == 'cosine':
            recomender =  CosineRecommender(K=commit_size)
        else:
            recomender = ItemItemRecommender(K=commit_size)

        self.models['KNN' + ' ' + dist_method] = ImplicitItemKNNWrapperModel(recomender)
        self.models['KNN' + ' ' + dist_method].fit(self.dataset)
        self.recos['KNN' + ' ' + dist_method] = self.models['KNN' + ' ' + dist_method].recommend(
            users=self.rating[Columns.User].unique(),
            dataset=self.dataset,
            k=commit_size,
            filter_viewed=True,
        )
        #print(self.recos['KNN'].head(40))
        print('KNN ' + dist_method)
        #return 0

    def recs_Random(self, commit_size = 10):
        self.models['Random'] = RandomModel(random_state=42)
        self.models['Random'].fit(self.dataset)
        self.recos['Random'] = self.models['Random'].recommend(
            users=self.rating[Columns.User].unique(),
            dataset=self.dataset,
            k=commit_size,
            filter_viewed=True,
        )
        #print(self.recos['Random'].head(10))
        print('Random')
        #return 0
    def recs_Popular(self, commit_size = 10):
        self.models['Popular'] = PopularModel()
        self.models['Popular'].fit(self.dataset)
        self.recos['Popular'] = self.models['Popular'].recommend(
            users=self.rating[Columns.User].unique(),
            dataset=self.dataset,
            k=commit_size,
            filter_viewed=True,
        )
        print('Popular')
        #return 0
    def recs_ALS(self, commit_size = 10):
        als_model = AlternatingLeastSquares(
            factors=100,
            regularization=0.01,
            iterations=50,
            dtype='float32',
            use_gpu=False  # или True, если доступно
        )
        self.models['ALS'] = ImplicitALSWrapperModel(als_model, fit_features_together=False )
        self.models['ALS'].fit(self.dataset)
        users = self.rating[Columns.User].unique()
        self.recos['ALS'] = self.models['ALS'].recommend(
            users=users,
            dataset=self.dataset,
            k=commit_size,
            filter_viewed=True
        )
        print('ALS')
        #return 0

    def metrics(self, df_test, model_name = 'Random'):
        metrics_values = {}
        k = self.k
        metrics = {
            "prec@" + str(k): Precision(k=k),
            "recall@" + str(k): Recall(k=k),
            "novelty@" + str(k): MeanInvUserFreq(k=k),
            "serendipity@" + str(k): Serendipity(k=k),
            "ndcg": NDCG(k=k, log_base=3)
        }

        #metrics_values['prec'] = metrics['prec@1'].calc_per_user(reco=self.recos[model_name], interactions=df_test)
        #print(f"precision1: {metrics_values['prec@1']}")
        metrics_values['prec'] = metrics["prec@" + str(k)].calc_per_user(reco=self.recos[model_name], interactions=df_test)
        #print(f"precision10: {metrics_values['prec@10']}")
        metrics_values['recall'] = metrics["recall@" + str(k)].calc_per_user(reco=self.recos[model_name], interactions=df_test)
        #print(f"recall10: {metrics_values['recall@10']}")
        metrics_values['ndcg'] = metrics['ndcg'].calc_per_user(reco=self.recos[model_name], interactions=df_test)
        #print(f"ndcg: {metrics_values['ndcg']}")
        catalog = self.rating[Columns.Item].unique()
        metrics_values['serendipity'] = metrics["serendipity@" + str(k)].calc_per_user(reco=self.recos[model_name], interactions=df_test, prev_interactions=self.rating, catalog=catalog)
        metrics_values['novelty'] = metrics["novelty@" + str(k)].calc_per_user(reco=self.recos[model_name], prev_interactions=self.rating)

        self.recos[model_name] = self.recos[model_name].merge(df_test[[Columns.User, Columns.Item, Columns.Weight]],
                                      on=[Columns.User, Columns.Item],
                                      how='left')
        self.recos[model_name] = self.recos[model_name].merge(self.movies, on="item_id")
        #print(self.recos[model_name].head(40))
        #print(self.recos[model_name][Columns.Weight])
        self.recos[model_name][Columns.Weight] = self.recos[model_name][Columns.Weight].fillna(0)
        arr = np.linspace(1 / 4, 1, 4)
        quantiles = self.rating.groupby(Columns.User)[Columns.Weight].quantile(arr)
        quantiles = quantiles.unstack()
        qarr = []
        qarr_dic = {}
        for user in self.rating[Columns.User].unique():
            qarr_dic[user] = []
        for index, row in self.recos[model_name].iterrows():
            #print(row)
            quants = quantiles.loc[row[Columns.User]]

            v = 0.5 if row[Columns.Weight] > 0 else 0
            # print(v)
            for q in quants:
                if row[Columns.Weight] > q:
                    v += 1
            qarr.append(v)
            qarr_dic[row[Columns.User]].append(v)
        metrics_values['weighted prec'] = {}
        for user in self.rating[Columns.User].unique():
            metrics_values['weighted prec'][user] = np.mean(qarr_dic[user]) / 3
        self.recos[model_name]['weighted weight'] = qarr

        # print(f"serendipity10: {metrics_values['serendipity@10']}")
        #return metrics_values, self.recos[model_name][[Columns.User, 'title', Columns.Weight, 'weighted weight']]
        self.recos[model_name] = self.recos[model_name].rename(columns={
            'title':'рекомендации',
            Columns.Weight: 'реальная оценка'
        })
        return metrics_values, self.recos[model_name][[Columns.User, 'рекомендации', 'реальная оценка']]
