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

class PBF:
    def __init__(self, matrix, p, weights = None):
        #n - количество переменных
        self.vars = []
        self.p = p
        self.n = len(matrix)
        self.vals = np.zeros(self.n)
        if weights is None:
            self.weights = np.ones(len(matrix.T))
        else:
            self.weights = weights
        for i in range(self.n):
            self.vars.append(i+1)
        self.dic = {}
        self.inverted_flag = False

        # if 2*p < self.n:
        #     matrix = self.invert(matrix)
        #     self.p = self.n - p
        #print(self.vars)

        # self.terms = [[], []]
        # for i in range(self.n):
        #     for c in combinations(self.vars, i):
        #         self.terms[0].append(frozenset(c))
        #     print(i)
        # #print(self.terms)
        # self.dic = {}
        # for t in self.terms[0]:
        #     self.dic[t] = 0
        #print(self.dic)
        self.from_matrix(matrix)

    def invert(self, matrix):
        matrix = np.array(matrix)
        max_el = np.max(matrix)
        new_matrix = -matrix + max_el
        self.inverted_flag = True
        return new_matrix

    def add_coef(self, term, c):
        t = frozenset(term)
        if t in self.dic:
            self.dic[t] += c
        else:
            self.dic[t] = c
        #print(self.dic)
    def calc(self, vars):
        S = 0
        ones = {index + 1 for index, value in enumerate(vars) if value == 1}
        #print(ones)
        for term, coef in self.dic.items():
            if term <= ones:
                S += coef
        #print(S)
        return S
    # def calc(self):
    #     S = 0
    #     ones = {index + 1 for index, value in enumerate(self.vals) if value == 1}
    #     #print(ones)
    #     for term, coef in self.dic.items():
    #         if term <= ones:
    #             S += coef
    #     #print(S)
    #     return S
    def from_column(self, column, voter):
        pi = sorted(range(len(column)), key=lambda i: column[i])
        #print(pi)
        cur = 0
        setlist = []
        weight = self.weights[voter]
        for i in pi:
            #print('to', setlist, 'add', column[i] - cur)
            self.add_coef(setlist,  weight*(column[i] - cur))
            cur = column[i]
            setlist.append(i + 1)
        #print(setlist)
        #print(self.dic)
    def from_matrix(self, matrix):
        C = np.array(matrix)
        #Ct = C.T
        for voter, col in enumerate(C.T):
            self.from_column(col, voter)
    def to_matrix(self):
        pass
    def print(self):
        for term, coef in self.dic.items():
            if coef != 0:
                output = ''.join(f"y_{i}" for i in sorted(term))
                print(str(coef) + output + ' +')

    def truncate(self):
        for term, coef in self.dic.items():
            if len(term) > (self.n - self.p):
                self.dic[term] = 0
    def useless(self):
        useless_vars = set()
        used_vars = set()
        for term, coef in self.dic.items():
            if coef != 0:
                for i in term:
                    used_vars.add(i)
        #print(used_vars)
        for i in self.vars:
            if i not in used_vars:
                useless_vars.add(i)
        #print(useless_vars)
        self.dic = {k: v for k, v in self.dic.items() if v != 0}
        return useless_vars
    def approx(self, tol = 0.05):
        high = 0
        num_nonzero = 0
        for term, coef in self.dic.items():
            if coef != 0:
                num_nonzero += 1
                high += coef
        mc = high/num_nonzero
        #print('mean coef is', mc, 'for', high, 'terms in general')
        gg = 0
        for term, coef in self.dic.items():
            if coef != 0 and coef < tol*mc:
                gg += 1
                self.dic[term] = 0
        #print(gg, 'terms deleted for being almost 0')

def branch(current, prohibited, divider, B, curval):
    # curL = deepcopy(current)
    curL = current.copy()
    curL[divider] = 1
    # proR = deepcopy(prohibited)
    # proR.add(divider)
    divider += 1
    return [[curL, prohibited, divider, 0, curval], [current, prohibited + 1, divider, 1, curval]]

def bound(polynome, p, ff, current, prohibited, divider, B, curval):
    zeros_count = polynome.n - np.count_nonzero(current)
    if B == 0 and (zeros_count == p or ff):
        val = polynome.calc(current)
    else:
        val = curval
    return [current, prohibited, divider, B, val], (zeros_count == p or prohibited > p or divider == polynome.n), zeros_count == p

# def BnB(matrix, p, tol=0, depth=True):
#     polynome = PBF(matrix, p)
#     # if 2*p < len(matrix):
#     #     p = len(matrix) - p
#     polynome.truncate()
#     # polynome.print()
#     if tol > 0:
#         polynome.approx(tol=tol)
#     useless = polynome.useless()
#     #print(len(useless), 'бесполезных кандидатов')
#     while len(useless) > polynome.n - p:
#         #print(polynome.n, p)
#         temp_list = list(useless)
#         useless.remove(temp_list[-1])
#         #print(len(useless), 'теперь бесполезных')
#     #empty_useless_flag = len(useless) == 0
#
#     init_vars = np.zeros(len(matrix))
#     state = np.zeros(len(matrix))
#     divider = 0
#     lowest = polynome.calc(np.ones(len(matrix)))
#     highest = polynome.calc(np.ones(len(matrix)))
#     start = polynome.calc(init_vars)
#     #print(lowest, start)
#
#     queue = deque()
#     queue.append([init_vars, 0, 0, 0, start])
#     feasible_flag = False
#     total_iterations = 1200000 * len(matrix)
#     count = 0
#     with tqdm(total=total_iterations) as pbar:
#         while (count < total_iterations or lowest == highest) and len(queue) > 0 and lowest != start:
#             count += 1
#             curr = queue.popleft()
#             #print(count)
#             if (curr[2] + 1) in useless:
#                 curr[0][curr[2]] = 1
#                 curr[2] += 1
#                 #curr[1] += 1
#                 curr[3] = 0
#                 queue.appendleft(curr)
#                 #useless.remove(curr[2])
#                 #empty_useless_flag = (len(useless) == 0)
#                 #print(curr, useless)
#                 if polynome.n - np.count_nonzero(curr[0]) == p:
#                     feasible_flag = True
#                     stop_flag = True
#                     if curr[4] < lowest:
#                         lowest = curr[4]
#                         state = curr[0]
#                 continue
#             curr, stop_flag, feasible = bound(polynome, p, feasible_flag, *curr)
#             # print('ветвление номер: ', count, ', текущее состояние: ', curr, stop_flag, feasible)
#             if curr[4] >= lowest:
#                 stop_flag = True
#             if stop_flag:
#                 if feasible:
#                     feasible_flag = True
#                     if curr[4] < lowest:
#                         lowest = curr[4]
#                         state = curr[0]
#             else:
#                 if depth:
#                     queue.extendleft(reversed(branch(*curr)))
#                 else:
#                     queue.extend(branch(*curr))
#             # progress.update(task, completed=time.time() - start_time)
#             # progress.console.print(f"Минимальная сумма: {lowest}", style="green", end="\r")
#             # pbar.update(time.time() - iter_time)
#             pbar.update(1)
#             # pbar.set_postfix({'lowest': lowest})
#             # print('очередь: ', queue)
#     if polynome.inverted_flag:
#         state = 1 - state
#     #print(count, 'итераций заняла оптимизация')
#     #print('оптимум:', state)
#     #print('сумма расстояний:', lowest)
#     #print('минимальная:', start)
#     return state
def BnB(matrix, p, tol=0, depth=True, weights = None):
    polynome = PBF(matrix, p, weights)
    # if 2*p < len(matrix):
    #     p = len(matrix) - p
    polynome.truncate()
    # polynome.print()
    if tol > 0:
        polynome.approx(tol=tol)
    useless = polynome.useless()
    #print(len(useless), 'бесполезных кандидатов')
    while len(useless) > polynome.n - p:
        #print(polynome.n, p)
        temp_list = list(useless)
        useless.remove(temp_list[-1])
        #print(len(useless), 'теперь бесполезных')
    #empty_useless_flag = len(useless) == 0

    init_vars = np.zeros(len(matrix))
    state = np.zeros(len(matrix))
    divider = 0
    lowest = polynome.calc(np.ones(len(matrix)))
    highest = polynome.calc(np.ones(len(matrix)))
    start = polynome.calc(init_vars)
    #print(lowest, start)

    queue = deque()
    queue.append([init_vars, 0, 0, 0, start])
    feasible_flag = False
    total_iterations = 1200000 * len(matrix)
    count = 0
    with tqdm(total=total_iterations) as pbar:
        while (count < total_iterations or lowest == highest) and len(queue) > 0 and lowest != start:
            count += 1
            curr = queue.popleft()
            #print(count)
            if (curr[2] + 1) in useless:
                curr[0][curr[2]] = 1
                curr[2] += 1
                #curr[1] += 1
                curr[3] = 0
                queue.appendleft(curr)
                #useless.remove(curr[2])
                #empty_useless_flag = (len(useless) == 0)
                #print(curr, useless)
                if polynome.n - np.count_nonzero(curr[0]) == p:
                    feasible_flag = True
                    stop_flag = True
                    if curr[4] < lowest:
                        lowest = curr[4]
                        state = curr[0]
                continue
            curr, stop_flag, feasible = bound(polynome, p, feasible_flag, *curr)
            # print('ветвление номер: ', count, ', текущее состояние: ', curr, stop_flag, feasible)
            if curr[4] >= lowest:
                stop_flag = True
            if stop_flag:
                if feasible:
                    feasible_flag = True
                    if curr[4] < lowest:
                        lowest = curr[4]
                        state = curr[0]
            else:
                if depth:
                    queue.extendleft(reversed(branch(*curr)))
                else:
                    queue.extend(branch(*curr))
            # progress.update(task, completed=time.time() - start_time)
            # progress.console.print(f"Минимальная сумма: {lowest}", style="green", end="\r")
            # pbar.update(time.time() - iter_time)
            pbar.update(1)
            # pbar.set_postfix({'lowest': lowest})
            # print('очередь: ', queue)
    if polynome.inverted_flag:
        state = 1 - state
    #print(count, 'итераций заняла оптимизация')
    #print('оптимум:', state)
    #print('сумма расстояний:', lowest)
    #print('минимальная:', start)
    return state