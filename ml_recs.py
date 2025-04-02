from os import environ

from implicit.nearest_neighbours import TFIDFRecommender
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

from lightfm import LightFM
from rectools.metrics import Precision, Recall, MAP, calc_metrics, MeanInvUserFreq, Serendipity, NDCG
sns.set_theme(style="whitegrid")

class Recommend():
    def __init__(self, rating, movies):
        self.dataset = Dataset.construct(rating)
        self.rating = rating
        self.movies = movies
        self.models = {}
        self.recos = {}
        #model1 = ImplicitALSWrapperModel()



    def recs_KNN(self, user_id, commit_size = 10):
        self.models['KNN'] = ImplicitItemKNNWrapperModel(TFIDFRecommender(K=commit_size))
        self.models['KNN'].fit(self.dataset)
        self.recos['KNN'] = self.models['KNN'].recommend(
            users=self.rating[Columns.User].unique(),
            dataset=self.dataset,
            k=commit_size,
            filter_viewed=True,
        )
        print(self.recos['KNN'].head(40))
        user_viewed = self.rating.query("user_id == @user_id").merge(self.movies, on="item_id")
        #print(user_viewed.query("weight > 0"))
        user_recos = self.recos['KNN'].query("user_id == @user_id").merge(self.movies, on="item_id")
        print(user_recos.head(10))
    def recs_Random(self, user_id, commit_size = 10):
        self.models['Random'] = RandomModel(random_state=42)
        self.models['Random'].fit(self.dataset)
        self.recos['Random'] = self.models['Random'].recommend(
            users=self.rating[Columns.User].unique(),
            dataset=self.dataset,
            k=commit_size,
            filter_viewed=True,
        )
        #print(self.recos['Random'].head(10))
        user_viewed = self.rating.query("user_id == @user_id").merge(self.movies, on="item_id")
        #print(user_viewed.query("weight > 6"))
        user_recos = self.recos['Random'].query("user_id == @user_id").merge(self.movies, on="item_id")
        print(user_recos.sort_values("rank"))
    def recs_Popular(self, user_id, commit_size = 10):
        self.models['Popular'] = PopularModel()
        self.models['Popular'].fit(self.dataset)
        self.recos['Popular'] = self.models['Popular'].recommend(
            users=self.rating[Columns.User].unique(),
            dataset=self.dataset,
            k=commit_size,
            filter_viewed=True,
        )
        #print(self.recos['Popular'].head(10))
        user_viewed = self.rating.query("user_id == @user_id").merge(self.movies, on="item_id")
        #print(user_viewed.query("weight > 6"))
        user_recos = self.recos['Popular'].query("user_id == @user_id").merge(self.movies, on="item_id")
        print(user_recos.head(10))
    # def recs_ALSW(self, commit_size):
    #     self.model_ALSW = ImplicitALSWrapperModel()

    def metrics(self, df_test, model_name = 'KNN'):
        metrics_values = {}
        metrics = {
            "prec@1": Precision(k=1),
            "prec@10": Precision(k=10),
            "recall@10": Recall(k=10),
            "novelty@10": MeanInvUserFreq(k=10),
            "serendipity@10": Serendipity(k=10),
            "ndcg": NDCG(k=10, log_base=3)
        }

        metrics_values['prec@1'] = metrics['prec@1'].calc_per_user(reco=self.recos[model_name], interactions=df_test)
        #print(f"precision1: {metrics_values['prec@1']}")
        metrics_values['prec@10'] = metrics['prec@10'].calc_per_user(reco=self.recos[model_name], interactions=df_test)
        #print(f"precision10: {metrics_values['prec@10']}")
        metrics_values['recall@10'] = metrics['recall@10'].calc_per_user(reco=self.recos[model_name], interactions=df_test)
        #print(f"recall10: {metrics_values['recall@10']}")
        metrics_values['ndcg'] = metrics['ndcg'].calc_per_user(reco=self.recos[model_name], interactions=df_test)
        #print(f"ndcg: {metrics_values['ndcg']}")
        catalog = self.rating[Columns.Item].unique()
        metrics_values['serendipity@10'] = metrics['serendipity@10'].calc_per_user(reco=self.recos[model_name], interactions=df_test, prev_interactions=self.rating, catalog=catalog)
        #print(f"serendipity10: {metrics_values['serendipity@10']}")
        return metrics_values


