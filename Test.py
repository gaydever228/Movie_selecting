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

class Test(election):
    def __init__(self, its, V, C, commit_size = 0, gen = True, distrV = 'normal', distrC = 'normal', boundV = 1, boundC = 1, matrix = None):
        super().__init__(V, C, commit_size, gen, distrV, distrC, boundV, boundC, matrix)
        self.iterations = its
    def test_BnB_time(self):
        params = {'k': [], 'tol': [], 'depth': [True, False]}
        self.Scores = {'BnB_depth': {'Score': [], 'Cost': []}, 'BnB_width': {'Score': [], 'Cost': []}, 'SNTV': {'Score': [], 'Cost': []}, 'alpha': []}
        for i in range(1, self.iterations):
            params['k'].append(i*self.C//self.iterations)
            self.Scores['alpha'].append(i/self.iterations)
            # params['tol'].append((i-1)/self.iterations)
        time_lists = {True: [], False: []}
        for dep in params['depth']:
            test_str = 'BnB_' + dep*'depth' + (1-dep)*'width'
            for k in params['k']:
                print(str(k) + '/' + str(self.C))
                print(test_str)
                self.k = k
                start_time = time.time()
                self.Scores[test_str]['Score'].append(self.BnB_rule(level = 0, depth = dep, draw_name = test_str + '_' + str(k/self.C)))
                time_lists[dep].append(time.time() - start_time)
                self.Scores[test_str]['Cost'].append(self.Cost)
            fig, ax = plt.subplots()
            # figure(figsize=(16, 12), dpi=80)
            ax.scatter(self.Scores['alpha'], self.Scores[test_str]['Score'], c='#0101dd', label='score(k/n)')
            #plt.show()
            fig.set_size_inches(10, 10)
            fig.savefig('score(alpha)_' + test_str + '.png', dpi=300)
            plt.close(fig)
        SNTV_time = []
        test_str = 'SNTV'
        for k in params['k']:
            print('SNTV: ' + str(k) + '/' + str(self.C))
            self.k = k
            start_time = time.time()
            self.Scores[test_str]['Score'].append(self.SNTV_rule(draw_name = test_str + '_' + str(k / self.C)))
            SNTV_time.append(time.time() - start_time)
            self.Scores[test_str]['Cost'].append(self.Cost)
        fig, ax = plt.subplots()
        # figure(figsize=(16, 12), dpi=80)
        ax.scatter(self.Scores['alpha'], self.Scores[test_str]['Score'], c='#0101dd', label='score(k/n)')
        # plt.show()
        fig.set_size_inches(10, 10)
        fig.savefig('score(alpha)_' + test_str + '.png', dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots()
        # figure(figsize=(16, 12), dpi=80)
        #ax.scatter(self.time_score_lists[True][2], self.time_score_lists[True][0], c='#56b100', label='depth')
        plt.plot(self.Scores['alpha'], time_lists[True], marker='^', mfc='r', mec='r', ms=6, ls='-', c='#56b100', lw=2, label='depth')
        #ax.scatter(self.time_score_lists[False][2], self.time_score_lists[False][0], c='#9867cf', label='width')
        plt.plot(self.Scores['alpha'], time_lists[False], marker = 'o', mfc = 'y', mec = 'y', ms = 6, c='#9867cf', lw=2, label='width')
        plt.plot(self.Scores['alpha'], SNTV_time, marker='.', mfc='k', mec='y', ms=6, c='#ffad0a', lw=2, label='SNTV')
        plt.xlabel("alpha")
        plt.ylabel("time, s")
        plt.legend()
        #plt.show()
        fig.set_size_inches(10, 10)
        fig.savefig('time(alpha).png', dpi=300)
        plt.close(fig)
        for s in ['Score', 'Cost']:
            fig, ax = plt.subplots()
            plt.plot(self.Scores['alpha'], self.Scores['BnB_depth'][s], marker='^', mfc='r', mec='r', ms=6, ls='-', c='#56b100', lw=2,
                     label='depth')
            plt.plot(self.Scores['alpha'], self.Scores['BnB_width'][s], marker='o', mfc='y', mec='y', ms=6, c='#9867cf', lw=2,
                     label='width')
            plt.plot(self.Scores['alpha'], self.Scores['SNTV'][s], marker='.', mfc='k', mec='y', ms=6, c='#ffad0a', lw=2, label='SNTV')
            plt.xlabel("alpha")
            plt.ylabel(s)
            plt.legend()
            # plt.show()
            fig.set_size_inches(10, 10)
            fig.savefig(s + '_Test(alpha).png', dpi=300)
            plt.close(fig)

    def test_BnB_tol(self):
        params = {'k': [], 'tol': []}
        for i in range(1, self.iterations + 1):
            params['k'].append(i * self.C // self.iterations)
            params['tol'].append((i - 1) / self.iterations)
        time_score_tol = []

