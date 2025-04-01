from os import environ

from implicit.nearest_neighbours import TFIDFRecommender
from pandas.core.interchange.dataframe_protocol import ColumnNullType

from main import raiting1

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
from rectools.metrics import MAP, calc_metrics, MeanInvUserFreq, Serendipity
sns.set_theme(style="whitegrid")

class Recommend():
    def __init__(self, rating):
        self.dataset = Dataset.construct(rating)
        #model1 = ImplicitALSWrapperModel()
        model2 = ImplicitItemKNNWrapperModel(TFIDFRecommender(K=10))
        model2.fit(self.dataset)
        recos = model2.recommend(
            users = rating[Columns.User].unique(),
            dataset = self.dataset,
            k=10,
            filter_viewed=True,
        )

