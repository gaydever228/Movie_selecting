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
import math
from random import sample
import random

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

class Reccomend(election):
    def __init__(self, headers, V = 100, C = 100, commit_size = 10, gen = False, distrV = 'normal', distrC = 'normal', boundV = 1, boundC = 1, Vote_matrix = None, raiting = None, degrees = 3, remove_rate = 1, bad_percent = 10):
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

    def App_Sets(self):
        self.approval_sets = {}
        for pos, cans in enumerate(self.VoteLists):
            deg = int((pos/self.C)//(1/self.degrees))
            #print(type(deg))
            if deg not in self.approval_sets:
                self.approval_sets[deg] = {}
            for v, c in enumerate(cans):
                c_int = int(c)
                if c_int in self.approval_sets[deg]:
                    self.approval_sets[deg][c_int].add(v)
                else:
                    self.approval_sets[deg][c_int] = set()
                    self.approval_sets[deg][c_int].add(v)
            for c in range(self.C):
                #c_int = int(c)
                if c not in self.approval_sets[deg]:
                    self.approval_sets[deg][c] = set()
        #print(self.approval_sets)


    def App_Sets_from_raiting3(self, raiting):
        self.degrees = 3
        self.approval_sets = {}
        self.voter_approval_sets = {}
        for i in range(self.degrees):
            self.approval_sets[i] = {}
            self.voter_approval_sets[i] = {}
            for voter in range(self.V):
                self.voter_approval_sets[i][voter] = set()
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
            print(voter, len(valid_raiting))
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
                    self.voter_approval_sets[0][voter].add(candidate)

                elif rate <= voters_dic[voter]['mid_low']:
                    self.approval_sets[2][candidate].add(voter)
                    self.voter_approval_sets[2][voter].add(candidate)
                elif not np.isnan(rate):
                    self.approval_sets[1][candidate].add(voter)
                    self.voter_approval_sets[1][voter].add(candidate)
        #print(self.approval_sets)
    def App_Sets_from_raiting10(self, raiting):
        self.degrees = 10
        self.approval_sets = {}
        self.voter_approval_sets = {}
        for i in range(self.degrees):
            self.approval_sets[i] = {}
            self.voter_approval_sets[i] = {}
            for voter in range(self.V):
                self.voter_approval_sets[i][voter] = set()
        # считаю, что кандидаты - строки, а столбцы - избиратели
        # пока что считаю, что degree = 10, а оценка ставится от 1 до 10

        for candidate, candidate_rates in enumerate(raiting):
            for i in range(self.degrees):
                self.approval_sets[i][candidate] = set()
            for voter, rate in enumerate(candidate_rates):
                self.approval_sets[int(rate)][candidate].add(voter)
                self.voter_approval_sets[int(rate)][voter].add(candidate)

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
        print("выбираем комитет мощности", commit_size, "из", len(c_to_c), "кандидатов с помощью", len(c_to_v), "избирателей")
        #self.dist_matrix = np.delete(np.delete(self.cand_dist, list(all_c_to_v), axis=0), list(c_without_bad), axis=1)  # конструкция матрицы расстояний чисто для этого голосования
        self.dist_matrix = self.cand_dist[np.ix_(sorted(c_to_c), sorted(c_to_v))]
        candidates = np.array(self.headers)

        #self.candidates = [np.delete(candidates, list(c_to_v)), np.delete(candidates, list(c_to_v))]  # зачем....?
        self.candidates = [candidates[np.ix_(sorted(c_to_c))], candidates[np.ix_(sorted(c_to_c))]]
        self.dist_matrix = np.square(self.dist_matrix)
        print(np.shape(self.dist_matrix))
        self.C = len(c_to_c)
        self.V = len(c_to_v)
        self.k = commit_size
        self.decision = None
        self.Score = None

        self.add_matrices(self.dist_matrix)
        #print(self.dist_matrix, self.candidates[0], len(c_to_c), len(c_to_v))
        self.k = min(self.k, len(c_to_c))
        if rule == 'SNTV':
            print('SNTV:', self.SNTV_rule())
            print(self.Score)
        elif rule == 'BnB':
            print('BnB:', self.BnB_rule(tol = 0.7, level=2))
            print(self.Cost)
            for id in self.committee_id:
               print('BnB recommends', self.candidates[0][id])
    def recommendation_voting(self, voter_id, commit_size = 10, rule = 'SNTV', method = 'no_bad'):
        if method == 'no_bad':
            c_to_v = set()  # множество фильмов, которые будут избирателями
            c_to_c = set()  # множество фильмов, из которых будем выбирать
            for i in range(self.C):
                c_to_c.add(i)
            for i in range(self.degrees):
                c_to_v.update(self.voter_approval_sets[i][voter_id])
            # удаляю ещё для проверки
            # for _ in range(50):
            #    c_to_v.pop()

            c_to_c = c_to_c.difference(c_to_v)
            self.voting(c_to_c, c_to_v, commit_size, rule)
            for id in self.committee_id:
                print('SNTV recommends', self.candidates[0][id])

        elif method == 'remove_bad' or method =='series':
            c_to_v = set()  # множество фильмов, которые будут избирателями
            c_to_c = set()  # множество фильмов, из которых будем выбирать
            all_c_to_v = set()  # множество всех оценённых фильмов
            for i in range(self.degrees):
                all_c_to_v.update(self.voter_approval_sets[i][voter_id])
            for i in range(self.C):
                c_to_c.add(i)
            print("количество оценённых фильмов", len(all_c_to_v))
            print("всего фильмов", len(c_to_c))
            c_to_v.update(self.voter_approval_sets[2][voter_id])  # избиратели - "плохие" фильмы
            bad_voters = self.headers[np.ix_(sorted(c_to_v))]
            c_to_c = c_to_c.difference(all_c_to_v)
            print(self.bad_percent, "% плохих фильмов", len(c_to_v))
            print("кандидатов", len(c_to_c))
            self.voting(c_to_c, c_to_v, commit_size * self.remove_rate)
            print('anti-reccomendations are:')
            for id in self.committee_id:
                print(self.candidates[0][id])
                index = np.where(np.array(self.headers) == self.candidates[0][id])
                #print(index[0][0])
                c_to_c.remove(index[0][0])
                nearest = np.argmin(self.dist_matrix[id, :])
                print("near to", bad_voters[nearest], self.dist_matrix[id, nearest])
            #c_to_v = all_c_to_v.difference(c_to_v)  # множество фильмов, которые будут избирателями
            c_to_v = set()
            c_to_v.update(self.voter_approval_sets[0][voter_id])
            if method == 'series':
                current_commit_size = max(int(len(c_to_c) * 0.75), commit_size)
                i = 1
                while current_commit_size > commit_size:
                    self.voting(c_to_c, c_to_v, current_commit_size)
                    print("step %d:" % i)
                    c_to_c = set()
                    for id in self.committee_id:
                        #print(self.candidates[0][id])
                        index = np.where(np.array(self.headers) == self.candidates[0][id])
                        #print(index[0][0])
                        c_to_c.add(index[0][0])
                    i += 1
                    current_commit_size = max(int(len(c_to_c) * 0.75), commit_size)
            voters = self.headers[np.ix_(sorted(c_to_v))]
            self.voting(c_to_c, c_to_v, commit_size)
            recos_list = []
            print('reccomendations are:')
            for id in self.committee_id:
                print(self.candidates[0][id])
                index = np.where(np.array(self.headers) == self.candidates[0][id])
                recos_list.append([voter_id, index[0][0], self.candidates[0][id]])
                nearest = np.nanargmin(self.dist_matrix[id, :])
                print("он близок к", voters[nearest], self.dist_matrix[id, nearest])
            self.recos = pd.DataFrame(recos_list, columns=[Columns.User, Columns.Item, "title"])
    def metrics(self, df_test, df_train):
        metrics_values = {}
        metrics = {
            "prec@1": Precision(k=1),
            "prec@10": Precision(k=10),
            "recall@10": Recall(k=10),
            "novelty@10": MeanInvUserFreq(k=10),
            "serendipity@10": Serendipity(k=10),
            "ndcg": NDCG(k=10, log_base=3)
        }

        metrics_values['prec@1'] = metrics['prec@1'].calc_per_user(reco=self.recos, interactions=df_test)
        print(f"precision1: {metrics_values['prec@1']}")
        metrics_values['prec@10'] = metrics['prec@10'].calc_per_user(reco=self.recos, interactions=df_test)
        print(f"precision10: {metrics_values['prec@10']}")
        metrics_values['recall@10'] = metrics['recall@10'].calc_per_user(reco=self.recos,
                                                                         interactions=df_test)
        print(f"recall10: {metrics_values['recall@10']}")
        metrics_values['ndcg'] = metrics['ndcg'].calc_per_user(reco=self.recos, interactions=df_test)
        print(f"ndcg: {metrics_values['ndcg']}")
        catalog = df_train[Columns.Item].unique()
        metrics_values['serendipity@10'] = metrics['serendipity@10'].calc_per_user(reco=self.recos,
                                                                                   interactions=df_test,
                                                                                   prev_interactions=df_train,
                                                                                   catalog=catalog)
        print(f"serendipity10: {metrics_values['serendipity@10']}")
        return metrics_values
        # elif method == 'series':
        #     #current_commit = 0.75*self.C
        #     c_to_v = set()  # множество фильмов, которые будут избирателями
        #     c_to_c = set()  # множество фильмов, из которых будем выбирать
        #     all_c_to_v = set()  # множество всех оценённых фильмов
        #     for i in range(self.degrees):
        #         all_c_to_v.update(self.voter_approval_sets[i][voter_id])
        #     candidates = self.headers
        #     for i in range(self.C):
        #         c_to_c.add(i)
        #     print("all rated", len(all_c_to_v))
        #     print("cands", len(c_to_c))
        #     c_to_v.update(self.voter_approval_sets[2][voter_id])  # избиратели - "плохие" фильмы
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












