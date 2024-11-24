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
    def __init__(self, V = 100, C = 100, commit_size = 10, gen = True, distrV = 'normal', distrC = 'normal', boundV = 1, boundC = 1, Vote_matrix = None, degrees = 1):
        if Vote_matrix is not None:
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
            print(self.VoteLists)
            self.real_cand_dist = np.zeros((self.C, self.C))
            #print(self.candidates)
            for i in range(self.C):
                for j in range(self.C):
                    self.real_cand_dist[i][j] = np.sqrt((self.candidates[0][i] -self.candidates[0][j])**2 + (self.candidates[1][i] -self.candidates[1][j])**2 )
        self.degrees = degrees
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
        print(self.approval_sets)
    def Candidates_dists(self):
        self.cand_dist = np.zeros((self.C, self.C))
        self.cand_dist += self.degrees
        #print(self.cand_dist)
        s = np.ones((self.degrees, self.degrees))
        for i in range(self.degrees):
            for j in range(self.degrees):
                s[i][j] -= 2*abs(i - j)/(self.degrees - 1)
        print(s)
        for c_1 in range(self.C):
            for c_2 in range(self.C):
                sum = 0
                for i in range(self.degrees):
                    for j in range(self.degrees):
                        if len(self.approval_sets[i][c_1] | self.approval_sets[j][c_2]) > 0:
                            sum += s[i][j]*len(self.approval_sets[i][c_1] & self.approval_sets[j][c_2])/len(self.approval_sets[i][c_1] | self.approval_sets[j][c_2])
                            print('i:', i, 'j:', j, 'c_1:', c_1, 'c_2:', c_2, 'intersection:', self.approval_sets[i][c_1] & self.approval_sets[j][c_2], 'union:', self.approval_sets[i][c_1] | self.approval_sets[j][c_2])
                        else:
                            sum += 1
                self.cand_dist[c_1][c_2] -= sum

        print(self.cand_dist)




