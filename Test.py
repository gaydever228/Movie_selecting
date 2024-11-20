import numpy as np
from itertools import combinations
from copy import deepcopy
from collections import deque
# библиотеки для симуляции и отрисовки выборов
import time
import matplotlib.pyplot as plt
from matplotlib import cm
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

class Test(election):
    def __init__(self, its, V, C, commit_size = 0, gen = True, distrV = 'normal', distrC = 'normal', boundV = 1, boundC = 1, matrix = None):
        super().__init__(V, C, commit_size, gen, distrV, distrC, boundV, boundC, matrix)
        self.iterations = its
        self.Scores = {'alpha': []}
        self.params = {'k': []}
        for i in range(1, self.iterations):
            self.params['k'].append(i * self.C // self.iterations)
            self.Scores['alpha'].append(i/self.iterations)
    def test_BnB_time(self):
        self.params['depth'] = [True, False]
        self.Scores['BnB_depth'] = {'Score': [], 'Cost': []}
        self.Scores['BnB_width'] = {'Score': [], 'Cost': []}
        time_lists = {True: [], False: []}
        for dep in self.params['depth']:
            test_str = 'BnB_' + dep*'depth' + (1-dep)*'width'
            for k in self.params['k']:
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
        return time_lists
    def SNTV_test(self):
        SNTV_time = []
        self.Scores['SNTV'] = {'Score': [], 'Cost': []}
        test_str = 'SNTV'
        #print(self.params)
        for k in self.params['k']:
            print('SNTV: ' + str(k) + '/' + str(self.C))
            self.k = k
            start_time = time.time()
            self.Scores[test_str]['Score'].append(self.SNTV_rule(draw_name = test_str + '_' + str(k / self.C)))
            SNTV_time.append(time.time() - start_time)
            #print(type((time.time() - start_time)))
            #print(SNTV_time)
            self.Scores[test_str]['Cost'].append(self.Cost)
        fig, ax = plt.subplots()
        # figure(figsize=(16, 12), dpi=80)
        ax.scatter(self.Scores['alpha'], self.Scores[test_str]['Score'], c='#0101dd', label='score(k/n)')
        # plt.show()
        fig.set_size_inches(10, 10)
        fig.savefig('score(alpha)_' + test_str + '.png', dpi=300)
        plt.close(fig)
        return SNTV_time
    def STV_test(self):
        STV_time = []
        test_str = 'STV'
        self.Scores['STV'] = {'Score': [], 'Cost': []}
        for k in self.params['k']:
            print('STV: ' + str(k) + '/' + str(self.C))
            self.k = k
            start_time = time.time()
            self.Scores[test_str]['Score'].append(self.STV_rule(draw_name = test_str + '_' + str(k / self.C)))
            #print(type((time.time() - start_time)))
            STV_time.append(time.time() - start_time)
            self.Scores[test_str]['Cost'].append(self.Cost)
        #print("STV", STV_time)
        fig, ax = plt.subplots()
        # figure(figsize=(16, 12), dpi=80)
        ax.scatter(self.Scores['alpha'], self.Scores[test_str]['Score'], c='#0101dd', label='score(k/n)')
        # plt.show()
        fig.set_size_inches(10, 10)
        fig.savefig('score(alpha)_' + test_str + '.png', dpi=300)
        plt.close(fig)
        return STV_time
    def test_rules(self):
        time_lists = self.test_BnB_time()
        SNTV_time = self.SNTV_test()
        STV_time = self.STV_test()
        tol_time = self.BnB_tol_test()

        time_lists['SNTV'] = SNTV_time
        time_lists['STV'] = STV_time
        df_times = pd.DataFrame(time_lists)
        df_tols = pd.DataFrame(tol_time)
        df_times.to_csv("times.csv", index=False)
        df_tols.to_csv("tols.csv", index=False)

        Scores_to_df = deepcopy(self.Scores)
        Scores_to_df.pop('alpha')
        Scores_to_df['alpha'] = {'Score': self.Scores['alpha'], 'Cost': self.Scores['alpha']}
        df_scores = pd.DataFrame.from_dict(Scores_to_df, orient='index')
        df_scores = df_scores.T
        df_scores.to_csv('scores_costs.csv')

        #без различных порогов точности для BnB
        fig, ax = plt.subplots()
        # figure(figsize=(16, 12), dpi=80)
        #ax.scatter(self.time_score_lists[True][2], self.time_score_lists[True][0], c='#56b100', label='depth')
        plt.plot(self.Scores['alpha'], time_lists[True], marker='^', mfc='b', mec='b', ms=6, ls='-', c='#56b100', lw=2, label='depth')
        #ax.scatter(self.time_score_lists[False][2], self.time_score_lists[False][0], c='#9867cf', label='width')
        plt.plot(self.Scores['alpha'], time_lists[False], marker = '^', mfc = 'r', mec = 'r', ms = 6, c='#9867cf', lw=2, label='width')
        plt.plot(self.Scores['alpha'], SNTV_time, marker='.', mfc='k', mec='k', ms=6, c='#ffad0a', lw=2, label='SNTV')
        plt.plot(self.Scores['alpha'], STV_time, marker='.', mfc='k', mec='k', ms=6, c='#009ee3', lw=2, label='STV')
        plt.xlabel("alpha")
        plt.ylabel("time, s")
        plt.legend()
        #plt.show()
        fig.set_size_inches(10, 10)
        fig.savefig('time(alpha).png', dpi=300)
        plt.close(fig)
        # с различными порогами точности для BnB
        fig, ax = plt.subplots()
        # figure(figsize=(16, 12), dpi=80)
        # ax.scatter(self.time_score_lists[True][2], self.time_score_lists[True][0], c='#56b100', label='depth')
        plt.plot(self.Scores['alpha'], time_lists[True], marker='^', mfc='b', mec='b', ms=6, ls='-', c='#56b100', lw=2,
                 label='depth')
        # ax.scatter(self.time_score_lists[False][2], self.time_score_lists[False][0], c='#9867cf', label='width')
        plt.plot(self.Scores['alpha'], time_lists[False], marker='^', mfc='r', mec='r', ms=6, c='#9867cf', lw=2,
                 label='width')
        plt.plot(self.Scores['alpha'], SNTV_time, marker='.', mfc='k', mec='k', ms=6, c='#ffad0a', lw=2, label='SNTV')
        plt.plot(self.Scores['alpha'], STV_time, marker='.', mfc='k', mec='k', ms=6, c='#009ee3', lw=2, label='STV')

        cmap = cm.get_cmap("viridis", len(self.params['tol']))
        i = 0
        for tol in self.params['tol']:
            test_str = 'BnB_' + str(tol) + '_tol'
            plt.plot(self.Scores['alpha'], tol_time[tol], marker='o', mfc='g', mec='g', ms=1, c=cmap(i), lw=0.7, label='tol = ' + str(tol))
            i += 1

        plt.xlabel("alpha")
        plt.ylabel("time, s")
        plt.legend()
        # plt.show()
        fig.set_size_inches(10, 10)
        fig.savefig('time(alpha).png', dpi=300)
        plt.close(fig)

        # зависимости скора на одном графике
        for s in ['Score', 'Cost']:
            fig, ax = plt.subplots()
            plt.plot(self.Scores['alpha'], self.Scores['BnB_depth'][s], marker='^', mfc='b', mec='b', ms=6, ls='-', c='#56b100', lw=2, label='depth')
            plt.plot(self.Scores['alpha'], self.Scores['BnB_width'][s], marker = '^', mfc = 'r', mec = 'r', ms = 6, c='#9867cf', lw=2, label='width')
            plt.plot(self.Scores['alpha'], self.Scores['SNTV'][s], marker='.', mfc='k', mec='k', ms=6, c='#ffad0a', lw=2, label='SNTV')
            plt.plot(self.Scores['alpha'], self.Scores['STV'][s], marker='.', mfc='k', mec='k', ms=6, c='#009ee3', lw=2, label='STV')
            plt.xlabel("alpha")
            plt.ylabel(s)
            plt.legend()
            # plt.show()
            fig.set_size_inches(10, 10)
            fig.savefig(s + '_Test(alpha).png', dpi=300)
            plt.close(fig)

            fig, ax = plt.subplots()
            plt.plot(self.Scores['alpha'], self.Scores['SNTV'][s], marker='.', mfc='k', mec='k', ms=6, c='#ffad0a',
                     lw=2, label='SNTV')
            i = 0
            for tol in self.params['tol']:
                test_str = 'BnB_' + str(tol) + '_tol'
                plt.plot(self.Scores['alpha'], self.Scores[test_str][s], marker='^', mfc='r', mec='r', ms=1,
                         c=cmap(i), lw=0.7, label='tol = ' + str(tol))
                i += 1

            plt.xlabel("alpha")
            plt.ylabel(s)
            plt.legend()
            # plt.show()
            fig.set_size_inches(10, 10)
            fig.savefig('tol_' + s + 'test.png', dpi=300)
            plt.close(fig)

    def BnB_tol_test(self):
        print('запущен тест порогов')
        self.params['tol'] = []
        step = self.iterations//5
        for i in range(0, self.iterations + 1, step):
            self.params['tol'].append(i/(2*self.iterations))
        time_lists = {}
        for tol in self.params['tol']:
            test_str = 'BnB_' + str(tol) + '_tol'
            #print(test_str)
            self.Scores[test_str] = {'Score': [], 'Cost': []}
            time_lists[tol] = []
            for k in self.params['k']:
                print(str(k) + '/' + str(self.C))
                print(test_str)
                self.k = k
                start_time = time.time()
                self.Scores[test_str]['Score'].append(self.BnB_rule(tol = tol, level = 2, depth = True, draw_name = test_str + '_' + str(k/self.C)))
                time_lists[tol].append(time.time() - start_time)

                self.candidates = deepcopy(self.all_candidates)
                self.dist_matrix = deepcopy(self.all_dist_matrix)
                self.sorted_dist_matrix = deepcopy(self.all_sorted_dist_matrix)
                self.VoteLists = deepcopy(self.all_VoteLists)
                self.C = len(self.candidates[0])

                self.Scores[test_str]['Cost'].append(self.Cost)
            # fig, ax = plt.subplots()
            # # figure(figsize=(16, 12), dpi=80)
            # ax.scatter(self.Scores['alpha'], self.Scores[test_str]['Score'], c='#0101dd', label='score(k/n)')
            # #plt.show()
            # fig.set_size_inches(10, 10)
            # fig.savefig('score(alpha)_' + test_str + '.png', dpi=300)
            # plt.close(fig)
        for s in ['Score', 'Cost']:
            cmap = cm.get_cmap("viridis", len(self.params['tol']))
            fig, ax = plt.subplots()
            i = 0
            for tol in self.params['tol']:
                test_str = 'BnB_' + str(tol) + '_tol'
                plt.plot(self.Scores['alpha'], self.Scores[test_str][s], marker='.', mfc='k', mec='k', ms=2, c = cmap(i), lw=1, label='tol = ' + str(tol))
                i += 1
            plt.xlabel("alpha")
            plt.ylabel(s)
            plt.legend()
            # plt.show()
            fig.set_size_inches(10, 10)
            fig.savefig('tol_' + s + '(alpha).png', dpi=300)
            plt.close(fig)

        return time_lists

