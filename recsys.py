# библиотеки для BnB алгоритма поиска оптимальных кандидатов через pseudo bollean polynomes
import numpy as np
from itertools import combinations
from copy import deepcopy
from collections import deque
# библиотеки для симуляции и отрисовки выборов
import time
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import figure
from numpy.random import Generator, PCG64
from joblib import Parallel, delayed
import math
from random import sample
import random
from scipy.stats import pearsonr, spearmanr, kendalltau
from rectools import Columns
from rectools.metrics import Precision, Recall, MAP, calc_metrics, MeanInvUserFreq, Serendipity, NDCG


rng = Generator(PCG64())
import copy
from tqdm import tqdm
from rich.progress import Progress
import csv
from scipy.optimize import curve_fit
from election import election
from PBF import PBF, BnB, bound, branch
from Test import Test

class Recommend_new(election):
    def __init__(self, links, raiting, pivo, degrees = 4, remove_rate = 1, series_rate = 2, dist_method = 'jaccar',
                 weights = None, full_dist = False, ids_to_num = None, cand_dist_matrix = None):
        self.quantiles = None
        self.candidates = None
        self.recos = None
        self.pivo = pivo
        self.series_rate = series_rate
        self.approval_sets = {}
        self.user_approval_sets = {}
        self.U = raiting[Columns.User].nunique()
        self.I = raiting[Columns.Item].nunique()
        self.raiting = raiting
        self.links = links
        self.degrees = degrees
        self.headers = raiting[Columns.Item].unique()
        self.headers = self.headers.tolist()
        if ids_to_num is not None:
            self.id_to_num = ids_to_num
            #print(type(self.id_to_num))
            self.num_to_id = {v: k for k, v in self.id_to_num.items()}
        else:
            self.id_to_num = {}
        self.App_Sets(raiting)
        #exit()
        self.dist_method = dist_method
        self.full_dist = full_dist
        s = np.ones((self.degrees, self.degrees))
        for i in range(self.degrees):
            for j in range(self.degrees):
                s[i][j] -= 2 * abs(i - j) / (self.degrees - 1)
        self.s = s
        if full_dist and cand_dist_matrix is None:
            self.Candidates_dists()
        elif cand_dist_matrix is not None:
            self.cand_dist = cand_dist_matrix
            #print(type(self.cand_dist))
        self.remove_rate = remove_rate

    def App_Sets(self, raiting):
        arr = np.linspace(1/self.degrees, 1, self.degrees)
        #print("степеней", self.degrees, ", квантили:", arr)
        quantiles = raiting.groupby(Columns.User)[Columns.Weight].quantile(arr)
        quantiles = quantiles.unstack()
        self.quantiles = quantiles
        #print('quants',quantiles)
        step = 1
        for user in raiting[Columns.User].unique():
            #print("App_Sets, user", user, "Step", step,"/",self.U)
            self.user_approval_sets[user] = {}
            for d in range(self.degrees):
                self.user_approval_sets[user][d] = set()
            user_weights = raiting[raiting[Columns.User] == user]
            user_quants = quantiles.loc[user]
            step += 1
            #print(f"User {user}:")
            for idx, row in user_weights.iterrows():
                weight = row[Columns.Weight]
                item = int(row['item_id'])
                #print(f"  Item {row[Columns.Item]} (weight={weight}):")
                if item not in self.approval_sets:
                    self.approval_sets[item] = {}
                    for d in range(self.degrees):
                        self.approval_sets[item][d] = set()
                for d in range(self.degrees):
                    if weight <= user_quants.iloc[d]:
                        if item not in self.approval_sets:
                            self.approval_sets[item] = {}
                        self.approval_sets[item][d].add(user)
                        self.user_approval_sets[user][d].add(item)
                        break
    def jaccar_joblib(self, c_1_num, c_2_num, c_1, c_2):
        sum = 0
        s = self.s
        #print(c_1_num, c_2_num)
        for i in range(self.degrees):
            for j in range(self.degrees):
                if len(self.approval_sets[c_1][i] | self.approval_sets[c_2][j]) > 0:
                    sum += s[i][j] * len(self.approval_sets[c_1][i] & self.approval_sets[c_2][j]) / len(
                        self.approval_sets[c_1][i] | self.approval_sets[c_2][j])
                    # print('i:', i, 'j:', j, 'c_1:', c_1, 'c_2:', c_2, 'intersection:', self.approval_sets[i][c_1] & self.approval_sets[j][c_2], 'union:', self.approval_sets[i][c_1] | self.approval_sets[j][c_2])
                elif i == j and c_1 == c_2:
                    sum += 1
        dist = self.degrees - sum
        return c_1_num, c_2_num, dist
    def spearman_joblib(self, c_1_num, c_2_num, c_1, c_2):
        r1 = np.array(self.pivo[c_1].fillna(0))
        r2 = np.array(self.pivo[c_2].fillna(0))
        if np.all(r1 == r1[0]) or np.all(r2 == r2[0]):
            return c_1_num, c_2_num, 1
        rho, p = spearmanr(r1, r2)
        dist = 1 - rho
        return c_1_num, c_2_num, dist
    def spearman_hat_joblib(self, c_1_num, c_2_num, c_1, c_2):
        r1 = np.array(self.pivo[c_1].fillna(0))
        r2 = np.array(self.pivo[c_2].fillna(0))
        mask = (r1 > 0) & (r2 > 0)
        r1_hat = r1[mask]
        r2_hat = r2[mask]
        uc = len(r1_hat)
        if uc == 0:
            return c_1_num, c_2_num, 2
        if np.all(r1_hat == r1_hat[0]) or np.all(r2_hat == r2_hat[0]):
            return c_1_num, c_2_num, 1
        rho, p = spearmanr(r1_hat, r2_hat)
        dist = 1 - rho
        return c_1_num, c_2_num, dist
    def kendall_joblib(self, c_1_num, c_2_num, c_1, c_2):
        r1 = np.array(self.pivo[c_1].fillna(0))
        r2 = np.array(self.pivo[c_2].fillna(0))
        tau, p = kendalltau(r1, r2)
        dist = 1 - tau
        return c_1_num, c_2_num, dist
    def kendall_hat_joblib(self, c_1_num, c_2_num, c_1, c_2):
        r1 = np.array(self.pivo[c_1].fillna(0))
        r2 = np.array(self.pivo[c_2].fillna(0))
        mask = (r1 > 0) & (r2 > 0)
        r1_hat = r1[mask]
        r2_hat = r2[mask]
        uc = len(r1_hat)
        if uc == 0:
            return c_1_num, c_2_num, 2
        tau, p = kendalltau(r1_hat, r2_hat)
        dist = 1 - tau
        return c_1_num, c_2_num, dist
    def pearson_joblib(self, c_1_num, c_2_num, c_1, c_2):
        """Вычисляет корреляцию для одной пары"""
        #print(c_1_num, c_2_num)
        r1 = np.array(self.pivo[c_1].fillna(0))
        r2 = np.array(self.pivo[c_2].fillna(0))
        r1 = r1 - r1.mean()
        r2 = r2 - r2.mean()
        d = (np.sqrt((r1 @ r1) * (r2 @ r2)))
        # print(d)
        if d == 0:
            return c_1_num, c_2_num, 2
        cor = (r1 @ r2) / d
        dist = 1 - cor
        return c_1_num, c_2_num, dist
    def pearson_hat_joblib(self, c_1_num, c_2_num, c_1, c_2):
        """Вычисляет корреляцию для одной пары"""
        r1 = np.array(self.pivo[c_1].fillna(0))
        r2 = np.array(self.pivo[c_2].fillna(0))
        mask = (r1 > 0) & (r2 > 0)
        #print(mask)
        r1_hat = r1[mask]
        r2_hat = r2[mask]
        uc = len(r1_hat)
        #print(r1_hat, r2_hat)
        #print(uc)
        if uc <= 1:
            return c_1_num, c_2_num, 2
        r1_hat = r1_hat - r1_hat.mean()
        r2_hat = r2_hat - r2_hat.mean()
        d = (np.sqrt((r1_hat @ r1_hat) * (r2_hat @ r2_hat)))
        #print(d)
        if d == 0:
            return c_1_num, c_2_num, 2/np.sqrt(uc)
        cor = (r1_hat @ r2_hat) / d
        dist = (1 - cor) / np.sqrt(uc)
        return c_1_num, c_2_num, dist
    def cosine_joblib(self, c_1_num, c_2_num, c_1, c_2):
        r1 = np.array(self.pivo[c_1].fillna(0))
        r2 = np.array(self.pivo[c_2].fillna(0))

        dist = 1 - (r1 @ r2) / (np.sqrt((r1 @ r1) * (r2 @ r2)))
        return c_1_num, c_2_num, dist
    def cosine_hat_joblib(self, c_1_num, c_2_num, c_1, c_2):
        r1 = np.array(self.pivo[c_1].fillna(0))
        r2 = np.array(self.pivo[c_2].fillna(0))
        mask = (r1 > 0) & (r2 > 0)

        r1_hat = r1[mask]
        r2_hat = r2[mask]
        uc = len(r1_hat)
        if uc == 0:
            return c_1_num, c_2_num, 2
        cos = (r1_hat @ r2_hat) / (np.sqrt((r1_hat @ r1_hat) * (r2_hat @ r2_hat)))
        dist = (1 - cos) / np.sqrt(uc)
        return c_1_num, c_2_num, dist
    def distances(self):
        print('export')
        return self.cand_dist.tolist(), self.id_to_num

    def Candidates_dists(self):
        met_dic = {'cosine_hat': self.cosine_hat_joblib,
                   'cosine': self.cosine_joblib,
                   'jaccar': self.jaccar_joblib,
                   'pearson': self.pearson_joblib,
                   'pearson_hat': self.pearson_hat_joblib,
                   'spearman': self.spearman_joblib,
                   'spearman_hat': self.spearman_hat_joblib,
                   'kendall': self.kendall_joblib,
                   'kendall_hat': self.kendall_hat_joblib}

        self.cand_dist = np.zeros((self.I, self.I))
        #print(self.I)
        vyzov = met_dic[self.dist_method]
        for c_1_num, c_1 in enumerate(self.headers):
            self.id_to_num[c_1] = c_1_num
            #print(self.dist_method + ': ' + str(c_1_num) + '/' + str(self.I))
            for c_2_num, c_2 in enumerate(self.headers):
                _, _, dist = vyzov(c_1_num, c_2_num, c_1, c_2)
                self.cand_dist[c_1_num][c_2_num] = dist
    # def Candidates_dists_old(self, method = 'jaccar'):
    #     if method == 'jaccar':
    #         self.cand_dist = np.zeros((self.I, self.I))
    #         self.cand_dist += self.degrees
    #         #print(self.cand_dist)
    #         s = np.ones((self.degrees, self.degrees))
    #         for i in range(self.degrees):
    #             for j in range(self.degrees):
    #                 s[i][j] -= 2*abs(i - j)/(self.degrees - 1)
    #         #print(s)
    #         step = 1
    #         for c_1_num, c_1 in enumerate(self.headers):
    #             self.id_to_num[c_1] = c_1_num
    #             #print('item', c_1, 'number', c_1_num,'/',self.I)
    #             for c_2_num, c_2 in enumerate(self.headers):
    #                 sum = 0
    #                 for i in range(self.degrees):
    #                     for j in range(self.degrees):
    #                         if len(self.approval_sets[c_1][i] | self.approval_sets[c_2][j]) > 0:
    #                             sum += s[i][j]*len(self.approval_sets[c_1][i] & self.approval_sets[c_2][j])/len(self.approval_sets[c_1][i] | self.approval_sets[c_2][j])
    #                             #print('i:', i, 'j:', j, 'c_1:', c_1, 'c_2:', c_2, 'intersection:', self.approval_sets[i][c_1] & self.approval_sets[j][c_2], 'union:', self.approval_sets[i][c_1] | self.approval_sets[j][c_2])
    #                         elif i == j and c_1 == c_2:
    #                             sum += 1
    #                 self.cand_dist[c_1_num][c_2_num] -= sum
    #     elif self.dist_method == 'jaccar_p':
    #         self.cand_dist = np.zeros((self.I, self.I))
    #         for c_1_num, c_1 in enumerate(self.headers):
    #             for c_2_num, c_2 in enumerate(self.headers):
    #                 _, _, dist = self.jaccar_joblib(c_1_num, c_2_num, c_1, c_2)
    #                 self.cand_dist[c_1_num][c_2_num] = dist
    #     elif self.dist_method == 'pearson':
    #         self.cand_dist = np.ones((self.I, self.I))
    #         for c_1_num, c_1 in enumerate(self.headers):
    #             self.id_to_num[c_1] = c_1_num
    #             for c_2_num, c_2 in enumerate(self.headers):
    #                 #print(c_1_num, c_2_num)
    #                 r1 = np.array(self.pivo[c_1].fillna(0))
    #                 r2 = np.array(self.pivo[c_2].fillna(0))
    #                 r1 = r1 - r1.mean()
    #                 r2 = r2 - r2.mean()
    #                 d = (np.sqrt((r1 @ r1) * (r2 @ r2)))
    #                 # print(d)
    #                 if d == 0:
    #                     self.cand_dist[c_1_num][c_2_num] = 2
    #                 else:
    #                     self.cand_dist[c_1_num][c_2_num] -= (r1 @ r2) / d
    #     elif self.dist_method == 'pearson_p':
    #         self.cand_dist = np.zeros((self.I, self.I))
    #         for c_1_num, c_1 in enumerate(self.headers):
    #             for c_2_num, c_2 in enumerate(self.headers):
    #                 _, _, dist = self.pearson_joblib(c_1_num, c_2_num, c_1, c_2)
    #                 self.cand_dist[c_1_num][c_2_num] = dist
    #
    #     elif self.dist_method == 'kendall':
    #         self.cand_dist = np.ones((self.I, self.I))
    #         for c_1_num, c_1 in enumerate(self.headers):
    #             self.id_to_num[c_1] = c_1_num
    #             for c_2_num, c_2 in enumerate(self.headers):
    #                 #print(c_1_num, c_2_num)
    #                 r1 = np.array(self.pivo[c_1].fillna(0))
    #                 r2 = np.array(self.pivo[c_2].fillna(0))
    #                 tau, p = kendalltau(r1, r2)
    #                 self.cand_dist[c_1_num][c_2_num] -= tau
    #     elif self.dist_method == 'spearman':
    #         self.cand_dist = np.ones((self.I, self.I))
    #         for c_1_num, c_1 in enumerate(self.headers):
    #             self.id_to_num[c_1] = c_1_num
    #             for c_2_num, c_2 in enumerate(self.headers):
    #                 #print(c_1_num, c_2_num)
    #                 r1 = np.array(self.pivo[c_1].fillna(0))
    #                 r2 = np.array(self.pivo[c_2].fillna(0))
    #                 rho, p = spearmanr(r1, r2)
    #                 self.cand_dist[c_1_num][c_2_num] -= rho
    #     elif self.dist_method == 'kendall_hat':
    #         self.cand_dist = np.zeros((self.I, self.I))
    #         for c_1_num, c_1 in enumerate(self.headers):
    #             self.id_to_num[c_1] = c_1_num
    #             for c_2_num, c_2 in enumerate(self.headers):
    #                 r1 = np.array(self.pivo[c_1].fillna(0))
    #                 r2 = np.array(self.pivo[c_2].fillna(0))
    #                 mask = (r1 > 0) & (r2 > 0)
    #                 r1_hat = r1[mask]
    #                 r2_hat = r2[mask]
    #                 uc = len(r1_hat)
    #                 if uc == 0:
    #                     self.cand_dist[c_1_num][c_2_num] = 2
    #                 else:
    #                     tau, p = kendalltau(r1_hat, r2_hat)
    #                     self.cand_dist[c_1_num][c_2_num] = (1 - tau)/np.sqrt(uc)
    #     elif self.dist_method == 'spearman_hat':
    #         self.cand_dist = np.zeros((self.I, self.I))
    #         for c_1_num, c_1 in enumerate(self.headers):
    #             self.id_to_num[c_1] = c_1_num
    #             for c_2_num, c_2 in enumerate(self.headers):
    #                 r1 = np.array(self.pivo[c_1].fillna(0))
    #                 r2 = np.array(self.pivo[c_2].fillna(0))
    #                 mask = (r1 > 0) & (r2 > 0)
    #                 r1_hat = r1[mask]
    #                 r2_hat = r2[mask]
    #                 uc = len(r1_hat)
    #                 if uc == 0:
    #                     self.cand_dist[c_1_num][c_2_num] = 2
    #                 else:
    #                     rho, p = spearmanr(r1_hat, r2_hat)
    #                     self.cand_dist[c_1_num][c_2_num] = (1 - rho)/np.sqrt(uc)
    #     elif self.dist_method == 'cosine':
    #         self.cand_dist = np.ones((self.I, self.I))
    #         for c_1_num, c_1 in enumerate(self.headers):
    #             self.id_to_num[c_1] = c_1_num
    #             for c_2_num, c_2 in enumerate(self.headers):
    #                 r1 = np.array(self.pivo[c_1].fillna(0))
    #                 r2 = np.array(self.pivo[c_2].fillna(0))
    #                 self.cand_dist[c_1_num][c_2_num] -= (r1 @ r2) / (np.sqrt((r1 @ r1) * (r2 @ r2)))
    #     elif self.dist_method == 'pearson_hat':
    #         self.cand_dist = np.zeros((self.I, self.I))
    #         for c_1_num, c_1 in enumerate(self.headers):
    #             self.id_to_num[c_1] = c_1_num
    #             for c_2_num, c_2 in enumerate(self.headers):
    #                 r1 = np.array(self.pivo[c_1].fillna(0))
    #                 r2 = np.array(self.pivo[c_2].fillna(0))
    #                 mask = (r1 > 0) & (r2 > 0)
    #                 r1_hat = r1[mask]
    #                 r2_hat = r2[mask]
    #                 uc = len(r1_hat)
    #                 if uc <= 1:
    #                     self.cand_dist[c_1_num][c_2_num] = 2
    #                 else:
    #                     r1_hat = r1_hat - r1_hat.mean()
    #                     r2_hat = r2_hat - r2_hat.mean()
    #                     d = (np.sqrt((r1_hat @ r1_hat) * (r2_hat @ r2_hat)))
    #                     # print(d)
    #                     if d == 0:
    #                         self.cand_dist[c_1_num][c_2_num] = 2 / np.sqrt(uc)
    #                     else:
    #                         cor = (r1_hat @ r2_hat) / d
    #                         dist = (1 - cor) / np.sqrt(uc)
    #
    #                 self.cand_dist[c_1_num][c_2_num] = dist
    #     elif self.dist_method == 'cosine_hat':
    #         self.cand_dist = np.zeros((self.I, self.I))
    #         for c_1_num, c_1 in enumerate(self.headers):
    #             self.id_to_num[c_1] = c_1_num
    #             for c_2_num, c_2 in enumerate(self.headers):
    #                 r1 = np.array(self.pivo[c_1].fillna(0))
    #                 r2 = np.array(self.pivo[c_2].fillna(0))
    #                 mask = (r1 > 0) & (r2 > 0)
    #                 r1_hat = r1[mask]
    #                 r2_hat = r2[mask]
    #                 uc = len(r1_hat)
    #                 if uc == 0:
    #                     self.cand_dist[c_1_num][c_2_num] = 2
    #                 else:
    #                     cos = (r1_hat @ r2_hat) / (np.sqrt((r1_hat @ r1_hat) * (r2_hat @ r2_hat)))
    #                     self.cand_dist[c_1_num][c_2_num] = (1 - cos)/np.sqrt(uc)
    #
    #     #print(self.cand_dist)
    def nes_cand_dist(self, cands, voters):
        if self.full_dist:
            voters_nums = {self.id_to_num[id] for id in voters}
            cands_nums = {self.id_to_num[id] for id in cands}

            #print('cands',cands, cands_nums)
            #print('voters', voters, voters_nums)
            dist_matrix = self.cand_dist[np.ix_(sorted(cands_nums), sorted(voters_nums))]
            self.voters_list = []
            #print('dist', dist_matrix)
            #self.dist_matrix = np.square(dist_matrix)
            self.dist_matrix = dist_matrix
            cand_arr = []
            for num in sorted(cands_nums):
                cand_arr.append(self.num_to_id[num])
            for num in sorted(voters_nums):
                self.voters_list.append(self.num_to_id[num])
            self.candidates = [np.array(cand_arr), np.array(cand_arr)]
            self.weights = np.array([self.weights_dic[self.num_to_id[num]] for num in sorted(voters_nums)])
        else:
            met_dic = {'cosine_hat':self.cosine_hat_joblib,
                       'cosine':self.cosine_joblib,
                       'jaccar':self.jaccar_joblib,
                       'pearson':self.pearson_joblib,
                       'pearson_hat':self.pearson_hat_joblib,
                       'spearman':self.spearman_joblib,
                       'spearman_hat':self.spearman_hat_joblib,
                       'kendall':self.kendall_joblib,
                       'kendall_hat':self.kendall_hat_joblib}

            dist_matrix = np.zeros((len(cands), len(voters)))
            sorted_cands = sorted(cands)
            sorted_voters = sorted(voters)
            vyzov = met_dic[self.dist_method]
            for c_1_num, c_1 in enumerate(sorted_cands):
                for c_2_num, c_2 in enumerate(sorted_voters):
                    _, _, dist = vyzov(c_1_num, c_2_num, c_1, c_2)
                    dist_matrix[c_1_num][c_2_num] = dist
            self.dist_matrix = np.square(dist_matrix)
            cand_arr = np.zeros(len(cands), dtype=int)
            for c in sorted(cands):
                cand_arr[self.id_to_num[c]] = c
            self.candidates = [cand_arr, cand_arr]

    # def nes_cand_dist_parallel_shit(self, cands, voters):
    #     if self.full_dist:
    #         voters_nums = {self.id_to_num[id] for id in voters}
    #         cands_nums = {self.id_to_num[id] for id in cands}
    #         dist_matrix = self.cand_dist[np.ix_(sorted(cands_nums), sorted(voters_nums))]
    #         return dist_matrix
    #     met_dic = {'cosine_hat':{'function': self.cosine_hat_joblib},
    #                'cosine':{'function': self.cosine_joblib},
    #                'jaccar':{'function': self.jaccar_joblib},
    #                'pearson':{'function': self.pearson_joblib},
    #                'pearson_hat':{'function': self.pearson_hat_joblib},
    #                'spearman':{'function': self.spearman_joblib},
    #                'spearman_hat':{'function': self.spearman_hat_joblib},
    #                'kendall':{'function': self.kendall_joblib},
    #                'kendall_hat':{'function': self.kendall_hat_joblib}}
    #
    #     dist_matrix = np.zeros((len(cands), len(voters)))
    #     sorted_cands = sorted(cands)
    #     sorted_voters = sorted(voters)
    #     vyzov = met_dic[self.dist_method]
    #     tasks = []
    #     for c_1_num, c_1 in enumerate(sorted_cands):
    #         for c_2_num, c_2 in enumerate(sorted_voters):
    #             tasks.append((c_1_num, c_2_num, c_1, c_2))
    #     print(dist_matrix.shape)
    #     results = Parallel(n_jobs=-1)(
    #         delayed(**vyzov)(c_1_num, c_2_num, c_1, c_2)
    #         for c_1_num, c_2_num, c_1, c_2 in tasks
    #     )
    #     for c_1_num, c_2_num, dist in results:
    #         dist_matrix[c_1_num][c_2_num] = dist
    #
    #     return dist_matrix

    # def voting(self, c_to_c, c_to_v, commit_size, rule = 'SNTV'):
    #     #c_to_c - множество movieId!
    #     #print("избиратели: ", sorted(c_to_v), type(sorted(c_to_v)[0]))
    #     #print(len(c_to_v))
    #     #print("кандидаты: ", c_to_c)
    #     #print(len(c_to_c))
    #     #print("выбираем комитет мощности", commit_size, "из", len(c_to_c), "кандидатов с помощью", len(c_to_v), "избирателей")
    #     #self.dist_matrix = np.delete(np.delete(self.cand_dist, list(all_c_to_v), axis=0), list(c_without_bad), axis=1)  # конструкция матрицы расстояний чисто для этого голосования
    #     self.dist_matrix = self.cand_dist[np.ix_(sorted(c_to_c), sorted(c_to_v))]
    #     self.candidates = [np.array(sorted(c_to_c)), np.array(sorted(c_to_c))]
    #     self.dist_matrix = np.square(self.dist_matrix)
    #     #print(np.shape(self.dist_matrix))
    #     self.C = len(c_to_c)
    #     self.V = len(c_to_v)
    #     self.k = commit_size
    #     self.decision = None
    #     self.Score = None
    #
    #     self.add_matrices(self.dist_matrix)
    #     #print(self.dist_matrix, self.candidates[0], len(c_to_c), len(c_to_v))
    #     self.k = min(self.k, len(c_to_c))
    #     if rule == 'SNTV':
    #         self.SNTV_rule()
    #         #print('SNTV:', self.SNTV_rule())
    #         #print(self.Score)
    #     elif rule == 'BnB':
    #         self.BnB_rule(tol=0.7, level=2)
    #         #print('BnB:', self.BnB_rule(tol = 0.7, level=2))
    #         #print(self.Cost)
    #         #for id in self.committee_id:
    #         #   print('BnB recommends', self.candidates[0][id])
    def voting(self, cands, voters, commit_size, rule = 'SNTV'):

        #self.dist_matrix = self.cand_dist[np.ix_(sorted(cand_nums), sorted(voters_nums))]
        #self.candidates = [np.array(sorted(cands)), np.array(sorted(cands))]
        #print('CANDIDATES:')
        #for c in sorted(cands):
        #    print(c, self.links[c])
        #print('cands?', sorted(cands))
        # for v in sorted(voters):
        #     print(v, self.links[v])
        self.nes_cand_dist(cands, voters)
        #print('cands!', sorted(set(self.candidates[0])))
        # print(np.shape(self.dist_matrix))
        self.C = len(cands)
        self.V = len(voters)
        self.k = commit_size
        self.decision = None
        self.Score = None
        self.add_matrices_micro(self.dist_matrix)
        #print('cands!!!', sorted(set(self.candidates[0])))
        # print(self.dist_matrix, self.candidates[0], len(c_to_c), len(c_to_v))
        self.k = min(self.k, len(cands))
        #self.weights = weights
        if rule == 'SNTV':
            self.SNTV_rule()
            # print('SNTV:', self.SNTV_rule())
            #print(self.Score)
            #for id in self.committee_id:
                #print('SNTV recommends', self.candidates[0][id])
        elif rule == 'BnB':
            self.add_matrices(self.dist_matrix)
            self.BnB_rule(tol=1, level=2)
            # print('BnB:', self.BnB_rule(tol = 0.7, level=2))
            # print(self.Cost)
            # for id in self.committee_id:
            #   print('BnB recommends', self.candidates[0][id])
        elif rule == 'STV_basic':
            self.STV_basic()
            #print(self.Score)
        elif rule == 'STV_star':
            self.STV_star()
            #print('STV_star reccomends', self.Score)
    def recommendation_voting(self, user_id, commit_size=10, rule='SNTV', weighted = False):
        c_to_v = set()  # множество фильмов, которые будут избирателями
        all_c_to_v = set(
            self.raiting[self.raiting[Columns.User] == user_id][Columns.Item])  # множество всех оценённых фильмов
        #all_items_num = {self.id_to_num[id] for id in all_c_to_v}
        all_items_set = set(self.raiting[Columns.Item])  # множество вообще всех фильмов
        c_to_c = all_items_set - all_c_to_v  # множество фильмов, из которых будем выбирать
        if weighted:
            self.weights_dic = {}
            # voters_nums = {self.id_to_num[id] for id in c_to_v}
            # cands_nums = {self.id_to_num[id] for id in c_to_c}
            for voter in sorted(all_c_to_v):
                for d in range(1, self.degrees):
                    if voter in self.user_approval_sets[user_id][d]:
                        self.weights_dic[voter] =d
            for d in range(1, self.degrees):
                c_to_v.update(self.user_approval_sets[user_id][d])  # множество фильмов, которые будут избирателями


            #print('weights', self.weights)
        else:
            #print('anti rec')
            self.weights_dic = {}
            # voters_nums = {self.id_to_num[id] for id in c_to_v}
            # cands_nums = {self.id_to_num[id] for id in c_to_c}
            for voter in sorted(all_c_to_v):
                self.weights_dic[voter] = 1
            for d in range(self.degrees):
                c_to_v.update(self.user_approval_sets[user_id][d])  # избиратели - "плохие" фильмы
            #
            #
            # self.voting(c_to_c, c_to_v, commit_size * self.remove_rate, rule)
            #
            # for id in self.committee_id:
            #     c_to_c.remove(self.candidates[0][id])
            # #self.dist_matrix = np.delete(self.dist_matrix, self.committee_id, axis=0)
            #     #cands_nums.remove(id)
            #
            # c_to_v = all_c_to_v.difference(c_to_v)  # множество фильмов, которые будут избирателями


        #self.dist_matrix = self.nes_cand_dist(c_to_c, c_to_v)
        # если series_rate = 0, будет просто единичное голосование
        current_commit_size = max(min(int(commit_size*((4/3)**self.series_rate)), (3*len(c_to_c))//4), commit_size)
        #cands_nums = {self.id_to_num[id] for id in c_to_c}
        if current_commit_size > commit_size:
            self.voting(c_to_c, c_to_v, current_commit_size, rule)
            # print("step %d:" % i)
            c_to_c = {self.candidates[0][id] for id in self.committee_id}
        #self.dist_matrix = self.dist_matrix[self.committee_id]
        #cands_nums = set(self.committee_id)
        #current_commit_size = max(min(int(commit_size*((4/3)**self.series_rate)), (3*len(c_to_c))//4), commit_size)
        #i += 1
        # voters_nums = {self.id_to_num[id] for id in c_to_v}
        # cands_nums = {self.id_to_num[id] for id in c_to_c}
        self.voting(c_to_c, c_to_v, commit_size, rule)
        recos_list = []
        #print('reccomendations are:')
        i = 1
        for id in self.committee_id:
            nearest_id = np.nanargmin(self.dist_matrix[id, :])
            nearest = self.voters_list[nearest_id]
            recos_list.append([user_id, self.candidates[0][id], i, self.links[self.candidates[0][id]],
                               self.links[nearest], nearest])
            # print(self.candidates[0][id])


            # print("он близок к", voters[nearest], self.dist_matrix[id, nearest])
            i += 1
        self.recos = pd.DataFrame(recos_list, columns=[Columns.User, Columns.Item, Columns.Rank, "title", "nearest", "nearest_id"])
        #print(self.recos)
        return list(self.recos['title'])


    def metrics(self, df_test, df_train, user_id):
        metrics_values = {}
        k = self.k
        metrics = {
            "prec@" + str(k): Precision(k=k),
            "recall@" + str(k): Recall(k=k),
            "novelty@" + str(k): MeanInvUserFreq(k=k),
            "serendipity@" + str(k): Serendipity(k=k),
            "ndcg": NDCG(k=k, log_base=3)
        }
        catalog = df_train[Columns.Item].unique()
        #metric_values_warp = calc_metrics(metrics, reco=self.recos, interactions=df_test, prev_interactions=df_train, catalog=catalog)[user_id]
        metrics_values["prec"] = metrics["prec@" + str(k)].calc_per_user(reco=self.recos, interactions=df_test)[user_id]
        #print(f"precision10: {metrics_values['prec@10']}")
        metrics_values['recall'] = metrics["recall@" + str(k)].calc_per_user(reco=self.recos,
                                                                         interactions=df_test)[user_id]
        #print(f"recall10: {metrics_values['recall@10']}")
        metrics_values['ndcg'] = metrics['ndcg'].calc_per_user(reco=self.recos, interactions=df_test)[user_id]
        #print(f"ndcg: {metrics_values['ndcg']}")

        metrics_values['serendipity'] = metrics["serendipity@" + str(k)].calc_per_user(reco=self.recos,
                                                                                   interactions=df_test,
                                                                                   prev_interactions=df_train,
                                                                                   catalog=catalog)[user_id]
        metrics_values['novelty'] = metrics["novelty@" + str(k)].calc_per_user(reco=self.recos, prev_interactions=df_train)[user_id]
        self.recos = self.recos.merge(df_test[[Columns.User, Columns.Item, Columns.Weight]],
                           on=[Columns.User, Columns.Item],
                           how='left')

        self.recos = self.recos.merge(
            df_train[[Columns.User, Columns.Item, Columns.Weight]],
            left_on=[Columns.User, 'nearest_id'],
            right_on=[Columns.User, Columns.Item],
            how='left'
        )
        self.recos = self.recos.drop(columns=['item_id_y'])
        #print(self.recos.head(10))
        self.recos['weight_x'] = self.recos['weight_x'].fillna(0)

        self.recos['weight_y'] = self.recos['weight_y'].fillna(0)
        #print(recos)
        #median_value = df_train[df_train[Columns.User] == user_id][Columns.Weight].median()
        quants = self.quantiles.loc[user_id]

        qarr = []
        for row in self.recos['weight_x']:
            #print(row)
            v = 0.5 if row > 0 else 0
            #print(v)
            for q in quants:
                if row > q:
                    v+= 1
            qarr.append(v)
        self.recos['weighted weight'] = qarr
        metrics_values['weighted prec'] = np.mean(qarr)/(self.degrees - 1)
        #print(f"serendipity10: {metrics_values['serendipity@10']}")
        self.recos = self.recos.rename(columns={
            'title':'рекомендации',
            'weight_x': 'реальная оценка',
            'nearest': 'ближайший сосед',
            'weight_y': 'оценка соседа'
        })
        return metrics_values, self.recos[['рекомендации', 'реальная оценка', 'ближайший сосед', 'оценка соседа']]







class Reccomend(election):
    def __init__(self, headers, V = 100, C = 100, commit_size = 10, gen = False, distrV = 'normal', distrC = 'normal', boundV = 1, boundC = 1, Vote_matrix = None, raiting = None, degrees = 3, remove_rate = 1, bad_percent = 10):
        self.recos = None
        self.bad_percent = bad_percent
        if raiting is not None:
            self.V = V
            self.C = C
            if degrees == 10:
                self.App_Sets_from_raiting10(raiting)
            elif degrees == 3:
                self.App_Sets_from_raiting3(raiting)
            self.Candidates_dists()

        elif Vote_matrix is not None:
            self.VoteLists = Vote_matrix
            self.V = len(Vote_matrix[0])
            self.C = len(Vote_matrix)
            self.candidates = np.arange(self.C)
            self.voters = np.arange(self.V)
            self.dist_matrix = None
            self.sorted_dist_matrix = None
            self.decision = None
            self.Score = None
            self.k = commit_size
        else:
            super().__init__(V, C, commit_size, gen, distrV, distrC, boundV, boundC)
            #print(self.VoteLists)
            self.real_cand_dist = np.zeros((self.C, self.C))
            #print(self.candidates)
            for i in range(self.C):
                for j in range(self.C):
                    self.real_cand_dist[i][j] = np.sqrt((self.candidates[0][i] -self.candidates[0][j])**2 + (self.candidates[1][i] -self.candidates[1][j])**2 )
        self.degrees = degrees
        self.headers = np.array(headers)
        self.remove_rate = remove_rate

    def App_Sets(self, raiting):
        arr = np.arange(0, 1, 1/self.degrees)[1:]
        quantiles = raiting.groupby(Columns.User)[Columns.Weight].quantile(arr)
        quantiles = quantiles.unstack()
        self.approval_sets = {}
        self.user_approval_sets = {}
        for i in range(self.degrees):
            self.approval_sets[i] = {}
            self.user_approval_sets[i] = {}
            for voter in range(self.V):
                self.user_approval_sets[i][voter] = set()
        for user in raiting[Columns.User].unique():
            user_weights = raiting[raiting[Columns.User] == user]
            user_quants = quantiles.loc[user]

            print(f"User {user}:")
            for idx, row in user_weights.iterrows():
                weight = row[Columns.Weight]
                print(f"  Item {row[Columns.Item]} (weight={weight}):")

                if weight <= user_quants[0.25]:
                    print("    <= 25% quantile")
                elif weight <= user_quants[0.5]:
                    print("    Between 25% and 50% quantile")
                elif weight <= user_quants[0.75]:
                    print("    Between 50% and 75% quantile")
                else:
                    print("    > 75% quantile")
            print()

    def App_Sets_from_raiting3(self, raiting):
        self.degrees = 3
        self.approval_sets = {}
        self.user_approval_sets = {}
        for i in range(self.degrees):
            self.approval_sets[i] = {}
            self.user_approval_sets[i] = {}
            for voter in range(self.V):
                self.user_approval_sets[i][voter] = set()
        # считаю, что кандидаты - строки, а столбцы - избиратели
        # пока что считаю, что degree = 3
        voters_dic = {}
        valid_raiting = raiting[~np.isnan(raiting)]
        voters_means = np.nanmean(valid_raiting, axis = 0)
        voters_medians = np.nanmedian(valid_raiting, axis = 0)
        voters_quantile = np.nanpercentile(valid_raiting, self.bad_percent, axis=0) # bad_percent% худших кандидатов имеют рейтинг ниже этой отметки
        #print("средняя оценка:", voters_means, "\nмедиана оценки:", voters_medians)
        for voter, voter_values in enumerate(raiting.T):
            valid_raiting = voter_values[~np.isnan(voter_values)]
            #print(voter, len(valid_raiting))
            voters_means = np.nanmean(valid_raiting, axis=0)
            voters_medians = np.nanmedian(valid_raiting, axis=0)
            voters_quantile = np.nanpercentile(valid_raiting, self.bad_percent, axis=0)  # bad_percent% худших кандидатов имеют рейтинг ниже этой отметки
            voters_dic[voter] = {'mean': voters_means, 'median': voters_medians, 'mid_low': voters_quantile}
        for candidate, candidate_rates in enumerate(raiting):
            for i in range(self.degrees):
                self.approval_sets[i][candidate] = set()
            for voter, rate in enumerate(candidate_rates):
                #print(raiting[0][voter], raiting[candidate][0], rate)
                #print('voter', voter)
                #print("mediad:", voters_dic[voter]['median'])
                #print("mean:", voters_dic[voter]['mean'])
                if rate >= min(voters_dic[voter]['mean'], voters_dic[voter]['median']):
                    self.approval_sets[0][candidate].add(voter)
                    self.user_approval_sets[0][voter].add(candidate)

                elif rate <= voters_dic[voter]['mid_low']:
                    self.approval_sets[2][candidate].add(voter)
                    self.user_approval_sets[2][voter].add(candidate)
                elif not np.isnan(rate):
                    self.approval_sets[1][candidate].add(voter)
                    self.user_approval_sets[1][voter].add(candidate)
        #print(self.approval_sets)
    def App_Sets_from_raiting10(self, raiting):
        self.degrees = 10
        self.approval_sets = {}
        self.user_approval_sets = {}
        for i in range(self.degrees):
            self.approval_sets[i] = {}
            self.user_approval_sets[i] = {}
            for voter in range(self.V):
                self.user_approval_sets[i][voter] = set()
        # считаю, что кандидаты - строки, а столбцы - избиратели
        # пока что считаю, что degree = 10, а оценка ставится от 1 до 10

        for candidate, candidate_rates in enumerate(raiting):
            for i in range(self.degrees):
                self.approval_sets[i][candidate] = set()
            for voter, rate in enumerate(candidate_rates):
                self.approval_sets[int(rate)][candidate].add(voter)
                self.user_approval_sets[int(rate)][voter].add(candidate)

        #print(self.approval_sets)

    def Candidates_dists(self):
        self.cand_dist = np.zeros((self.C, self.C))
        self.cand_dist += self.degrees
        #print(self.cand_dist)
        s = np.ones((self.degrees, self.degrees))
        for i in range(self.degrees):
            for j in range(self.degrees):
                s[i][j] -= 2*abs(i - j)/(self.degrees - 1)
        #print(s)
        for c_1 in range(self.C):
            for c_2 in range(self.C):
                sum = 0
                for i in range(self.degrees):
                    for j in range(self.degrees):
                        if len(self.approval_sets[i][c_1] | self.approval_sets[j][c_2]) > 0:
                            sum += s[i][j]*len(self.approval_sets[i][c_1] & self.approval_sets[j][c_2])/len(self.approval_sets[i][c_1] | self.approval_sets[j][c_2])
                            #print('i:', i, 'j:', j, 'c_1:', c_1, 'c_2:', c_2, 'intersection:', self.approval_sets[i][c_1] & self.approval_sets[j][c_2], 'union:', self.approval_sets[i][c_1] | self.approval_sets[j][c_2])
                        elif i == j and c_1 == c_2:
                            sum += 1
                self.cand_dist[c_1][c_2] -= sum

        #print(self.cand_dist)
    def voting(self, c_to_c, c_to_v, commit_size, rule = 'SNTV'):
        #print("избиратели: ", sorted(c_to_v), type(sorted(c_to_v)[0]))
        #print(len(c_to_v))
        #print("кандидаты: ", c_to_c)
        #print(len(c_to_c))
        #print("выбираем комитет мощности", commit_size, "из", len(c_to_c), "кандидатов с помощью", len(c_to_v), "избирателей")
        #self.dist_matrix = np.delete(np.delete(self.cand_dist, list(all_c_to_v), axis=0), list(c_without_bad), axis=1)  # конструкция матрицы расстояний чисто для этого голосования
        self.dist_matrix = self.cand_dist[np.ix_(sorted(c_to_c), sorted(c_to_v))]
        candidates = np.array(self.headers)


        #self.candidates = [np.delete(candidates, list(c_to_v)), np.delete(candidates, list(c_to_v))]  # зачем....?
        self.candidates = [candidates[np.ix_(sorted(c_to_c))], candidates[np.ix_(sorted(c_to_c))]]
        self.dist_matrix = np.square(self.dist_matrix)
        #print(np.shape(self.dist_matrix))
        self.C = len(c_to_c)
        self.V = len(c_to_v)
        self.k = commit_size
        self.decision = None
        self.Score = None

        self.add_matrices_micro(self.dist_matrix)
        #print(self.dist_matrix, self.candidates[0], len(c_to_c), len(c_to_v))
        self.k = min(self.k, len(c_to_c))
        if rule == 'SNTV':
            self.SNTV_rule()
            #print('SNTV:', self.SNTV_rule())
            #print(self.Score)
        elif rule == 'BnB':
            self.add_matrices(self.dist_matrix)
            self.BnB_rule(tol=0.7, level=2)
            #print('BnB:', self.BnB_rule(tol = 0.7, level=2))
            #print(self.Cost)
            #for id in self.committee_id:
            #   print('BnB recommends', self.candidates[0][id])
    def recommendation_voting(self, voter_id, commit_size = 10, rule = 'SNTV', method = 'no_bad'):
        '''if method == 'no_bad':
            c_to_v = set()  # множество фильмов, которые будут избирателями
            c_to_c = set()  # множество фильмов, из которых будем выбирать
            for i in range(self.C):
                c_to_c.add(i)
            for i in range(self.degrees):
                c_to_v.update(self.user_approval_sets[i][voter_id])
            # удаляю ещё для проверки
            # for _ in range(50):
            #    c_to_v.pop()

            c_to_c = c_to_c.difference(c_to_v)
            self.voting(c_to_c, c_to_v, commit_size, rule)
            for id in self.committee_id:
                print('SNTV recommends', self.candidates[0][id])
'''
        if method == 'remove_bad' or method =='series':
            c_to_v = set()  # множество фильмов, которые будут избирателями
            c_to_c = set()  # множество фильмов, из которых будем выбирать
            all_c_to_v = set()  # множество всех оценённых фильмов
            for i in range(self.degrees):
                all_c_to_v.update(self.user_approval_sets[i][voter_id])
            for i in range(self.C):
                c_to_c.add(i)
            #print("количество оценённых фильмов", len(all_c_to_v))
            #print("всего фильмов", len(c_to_c))
            c_to_v.update(self.user_approval_sets[2][voter_id])  # избиратели - "плохие" фильмы
            bad_voters = self.headers[np.ix_(sorted(c_to_v))]
            c_to_c = c_to_c.difference(all_c_to_v)
            #print(self.bad_percent, "% плохих фильмов", len(c_to_v))
            #print("кандидатов", len(c_to_c))
            self.voting(c_to_c, c_to_v, commit_size * self.remove_rate, rule)
            #print('anti-reccomendations are:')
            for id in self.committee_id:
                #print(self.candidates[0][id])
                index = np.where(np.array(self.headers) == self.candidates[0][id])
                #print(index[0][0])
                c_to_c.remove(index[0][0])
                nearest = np.argmin(self.dist_matrix[id, :])
                #print("near to", bad_voters[nearest], self.dist_matrix[id, nearest])
            #c_to_v = all_c_to_v.difference(c_to_v)  # множество фильмов, которые будут избирателями
            c_to_v = set()
            c_to_v.update(self.user_approval_sets[0][voter_id])
            if method == 'series':
                current_commit_size = max(int(len(c_to_c) * 0.75), commit_size)
                i = 1
                while current_commit_size > commit_size:
                    self.voting(c_to_c, c_to_v, current_commit_size, rule)
                    #print("step %d:" % i)
                    c_to_c = set()
                    for id in self.committee_id:
                        #print(self.candidates[0][id])
                        index = np.where(np.array(self.headers) == self.candidates[0][id])
                        #print(index[0][0])
                        c_to_c.add(index[0][0])
                    i += 1
                    current_commit_size = max(int(len(c_to_c) * 0.75), commit_size)
            voters = self.headers[np.ix_(sorted(c_to_v))]
            self.voting(c_to_c, c_to_v, commit_size, rule)
            recos_list = []
            #print('reccomendations are:')
            i = 1
            for id in self.committee_id:
                #print(self.candidates[0][id])
                index = np.where(np.array(self.headers) == self.candidates[0][id])
                recos_list.append([voter_id, index[0][0], i, self.candidates[0][id]])
                nearest = np.nanargmin(self.dist_matrix[id, :])
                #print("он близок к", voters[nearest], self.dist_matrix[id, nearest])
                i += 1
            self.recos = pd.DataFrame(recos_list, columns=[Columns.User, Columns.Item, Columns.Rank, "title"])
            return list(self.recos['title'])
        return None

    def metrics(self, df_test, df_train, voter_id):
        metrics_values = {}
        metrics = {
            "prec@1": Precision(k=1),
            "prec@10": Precision(k=10),
            "recall@10": Recall(k=10),
            "novelty@10": MeanInvUserFreq(k=10),
            "serendipity@10": Serendipity(k=10),
            "ndcg": NDCG(k=10, log_base=3)
        }

        metrics_values['prec@1'] = metrics['prec@1'].calc_per_user(reco=self.recos, interactions=df_test)[voter_id]
        #print(f"precision1: {metrics_values['prec@1']}")
        metrics_values['prec@10'] = metrics['prec@10'].calc_per_user(reco=self.recos, interactions=df_test)[voter_id]
        #print(f"precision10: {metrics_values['prec@10']}")
        metrics_values['recall@10'] = metrics['recall@10'].calc_per_user(reco=self.recos,
                                                                         interactions=df_test)[voter_id]
        #print(f"recall10: {metrics_values['recall@10']}")
        metrics_values['ndcg'] = metrics['ndcg'].calc_per_user(reco=self.recos, interactions=df_test)[voter_id]
        #print(f"ndcg: {metrics_values['ndcg']}")
        catalog = df_train[Columns.Item].unique()
        metrics_values['serendipity@10'] = metrics['serendipity@10'].calc_per_user(reco=self.recos,
                                                                                   interactions=df_test,
                                                                                   prev_interactions=df_train,
                                                                                   catalog=catalog)[voter_id]
        #print(f"serendipity10: {metrics_values['serendipity@10']}")
        return metrics_values
        # elif method == 'series':
        #     #current_commit = 0.75*self.C
        #     c_to_v = set()  # множество фильмов, которые будут избирателями
        #     c_to_c = set()  # множество фильмов, из которых будем выбирать
        #     all_c_to_v = set()  # множество всех оценённых фильмов
        #     for i in range(self.degrees):
        #         all_c_to_v.update(self.user_approval_sets[i][voter_id])
        #     candidates = self.headers
        #     for i in range(self.C):
        #         c_to_c.add(i)
        #     print("all rated", len(all_c_to_v))
        #     print("cands", len(c_to_c))
        #     c_to_v.update(self.user_approval_sets[2][voter_id])  # избиратели - "плохие" фильмы
        #     c_to_c = c_to_c.difference(all_c_to_v)
        #     self.voting(c_to_c, c_to_v, commit_size * self.remove_rate)
        #     print('anti-reccomendations are:')
        #     for id in self.committee_id:
        #         print(self.candidates[0][id])
        #         index = np.where(np.array(self.headers) == self.candidates[0][id])
        #         print(index[0][0])
        #         c_to_c.remove(index[0][0])
        #     current_commit_size = max(len(c_to_c)*0.75, commit_size)
        #     c_to_v = all_c_to_v.difference(c_to_v)  # множество фильмов, которые будут избирателями
        #     i = 1
        #     while current_commit_size > commit_size:
        #         self.voting(c_to_c, c_to_v, current_commit_size)
        #         print("step %d:"%i)
        #         c_to_c = set()
        #         for id in self.committee_id:
        #             print(self.candidates[0][id])
        #             index = np.where(np.array(self.headers) == self.candidates[0][id])
        #             print(index[0][0])
        #             c_to_c.add(index[0][0])
        #         i += 1
        #         current_commit_size = max(len(c_to_c)*0.75, commit_size)
        #
        #     self.voting(c_to_c, c_to_v, commit_size)
        #     print('reccomendations are:')
        #     for id in self.committee_id:
        #         print(self.candidates[0][id])












