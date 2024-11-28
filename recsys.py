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

class Reccomend(election):
    def __init__(self, headers, V = 100, C = 100, commit_size = 10, gen = False, distrV = 'normal', distrC = 'normal', boundV = 1, boundC = 1, Vote_matrix = None, raiting = None, degrees = 1):
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
        self.headers = headers
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
        voters_means = np.nanmean(raiting, axis = 0)
        voters_medians = np.nanmedian(raiting, axis = 0)
        voters_quantile = np.nanpercentile(raiting, 10, axis=0)
        print(voters_means, voters_medians)
        for voter, voter_values in enumerate(raiting.T):
            voters_dic[voter] = {'mean': voters_means[voter], 'median': voters_medians[voter], 'mid_low': voters_quantile[voter]}
        for candidate, candidate_rates in enumerate(raiting):
            for i in range(self.degrees):
                self.approval_sets[i][candidate] = set()
            for voter, rate in enumerate(candidate_rates):
                if rate >= max(voters_dic[voter]['mean'], voters_dic[voter]['median']):
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
    def recommendation_voting(self, voter_id, commit_size = 10):
        c_to_v = set()
        c_to_c = set()
        candidates = self.headers
        for i in range(self.C):
            c_to_c.add(i)
        for i in range(self.degrees):
            c_to_v.update(self.voter_approval_sets[i][voter_id])
        # удаляю ещё для проверки
        #for _ in range(50):
        #    c_to_v.pop()

        c_to_c = c_to_c.difference(c_to_v)
        print(c_to_v)
        print(c_to_c)
        self.dist_matrix = np.delete(np.delete(self.cand_dist, list(c_to_v), axis = 0), list(c_to_c), axis = 1)
        self.candidates = [np.delete(candidates, list(c_to_v)), np.delete(candidates, list(c_to_v))]
        self.dist_matrix = np.square(self.dist_matrix)
        self.C = len(c_to_c)
        self.V = len(c_to_v)
        self.k = commit_size
        self.decision = None
        self.Score = None

        self.add_matrices(self.dist_matrix)
        print(self.dist_matrix, self.candidates[0], len(c_to_c), len(c_to_v))
        self.k = min(self.k, len(c_to_c))

        print('SNTV:', self.SNTV_rule())
        print(self.Cost)
        for id in self.committee_id:
            print('SNTV recommends', self.candidates[0][id])

        print('BnB:', self.BnB_rule(tol = 0.7, level=2))
        print(self.Cost)
        for id in self.committee_id:
            print('BnB recommends', self.candidates[0][id])









