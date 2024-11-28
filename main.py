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
import math
from random import sample
import random
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

# C = np.array([[7, 15, 10, 7, 10], [10, 17, 9, 11, 22], [16, 7, 6, 18, 14], [11, 7, 6, 12, 8]])
# Votes = np.argsort(C, axis=0)
# print(C)
# print(Votes)
# recs = Reccomend(Vote_matrix=Votes, degrees=2)
# recs.App_Sets()
# recs.Candidates_dists()

df = pd.read_csv('rate1.csv')
headers = list(df.columns)[1:]
print('headers', headers)
df = df.T
print(df.head(10))

raiting1 = np.array(df.values[1:], dtype = float)
print(raiting1)
print(type(raiting1[0, 0]), type(raiting1[1, 1]))


#for c in raiting1[1]:
#    print(c)
#    print(np.isnan(c))
#    print(c>10)
recs_test = Reccomend(headers, len(raiting1[0]), len(raiting1), degrees=3, raiting=raiting1)
recs_test.recommendation_voting(5, 10)
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
print('finish')
#print(testing_BnB.Scores)