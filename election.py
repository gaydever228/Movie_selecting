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
from PBF import PBF, BnB, bound, branch

class election:
    def __init__(self, V, C, commit_size = 0, gen = False, distrV = 'normal', distrC = 'normal', boundV = 1, boundC = 1, matrix = None, weights = None):
        self.decision = None
        self.Score = None
        self.V = V
        self.C = C
        self.k = commit_size
        if gen:
            self.generate(distrV, distrC, boundV, boundC)
            self.make_matrix()
            self.all_candidates = deepcopy(self.candidates)
            self.all_voters = deepcopy(self.voters)
            self.all_dist_matrix = deepcopy(self.dist_matrix)
            self.all_sorted_dist_matrix = deepcopy(self.sorted_dist_matrix)
            self.all_VoteLists = deepcopy(self.VoteLists)
            #self.sort_candidates()
        elif matrix is not None:
            self.dist_matrix = matrix
            self.sorted_dist_matrix = np.sort(matrix, axis=0)
            self.VoteLists = np.argsort(matrix, axis=0)
            print(self.VoteLists)
        if weights == None:
            self.weights = np.ones(V)
    def generate(self, distrV = 'normal', distrC = 'normal', boundV = 1, boundC = 1):
        # Генерируем кандидатов:
        if distrC == 'normal':
            self.candidates = boundC*rng.standard_normal((2, self.C))
        elif distrC == 'uniform':
            self.candidates = rng.uniform(-1*boundC, boundC, (2, self.C))
        # Генерируем избирателей:
        if distrV == 'normal':
            self.voters = boundV*rng.standard_normal((2, self.V))
        elif distrV == 'uniform':
            self.voters = rng.uniform(-1*boundV, boundV, (2, self.V))
    def make_matrix(self):
        distances = np.zeros((self.C, self.V))
        for i in range(self.C):
            for j in range(self.V):
                distances[i][j] = (self.voters[0][j]-self.candidates[0][i])**2 + (self.voters[1][j] - self.candidates[1][i])**2
        #self.sorted_dist_matrix = np.sort(distances, axis=0)
        self.add_matrices(distances)
    def add_matrices(self, distances):
        VoteLists = np.argsort(distances, axis=0)
        PS = np.zeros(self.C)
        for voter, e in enumerate(VoteLists.T):
            PS[e[0]] += self.weights[voter]
        #print(PS)
        sorted_by_PS = np.argsort(PS)
        sorted_matrix = []
        sorted_candidates = [[], []]
        for candidate in sorted_by_PS:
            sorted_matrix.append(distances[candidate])
            sorted_candidates[0].append(self.candidates[0][candidate])
            sorted_candidates[1].append(self.candidates[1][candidate])
        self.candidates  = np.array(sorted_candidates)
        self.dist_matrix = np.array(sorted_matrix)
        self.sorted_dist_matrix = np.sort(sorted_matrix, axis=0)
        self.VoteLists = np.argsort(sorted_matrix, axis=0)
    # def sort_candidates(self):
    #     PS = np.zeros(self.C)
    #     for e in self.VoteLists.T:
    #         PS[e[0]] += 1
    #     # в PS индексы - номера кандидатов, а значение - их SNTV рейтинг
    #     #print(PS)
    #     sorted_by_PS = np.argsort(PS)
    #     sorted_matrix = []
    #     sorted_candidates = []
    #     for candidate in sorted_by_PS:
    #         sorted_matrix.append(self.dist_matrix[candidate])
    #         sorted_candidates.append(self.candidates[candidate])
    #     self.candidates  = np.array(sorted_candidates)
    #     #sorted_matrix = np.array(sorted_matrix)
    #     #print(sorted_matrix.shape, self.dist_matrix.shape)
    #     #print(sorted_matrix, self.dist_matrix)
    #     self.dist_matrix = np.array(sorted_matrix)
    #     self.sorted_dist_matrix = np.sort(self.dist_matrix, axis=0)
    #     self.VoteLists = np.argsort(self.dist_matrix, axis=0)
    #     #return sorted_matrix
    def delete_max(self, level  = 2):
        #print('before:', self.dist_matrix)
        #print(self.VoteLists)
        anti_PS = np.zeros(self.C)
        PS = np.zeros(self.C)
        #CN = deepcopy(self.candidates)
        Cand = np.arange(self.C)
        for e in self.VoteLists.T:
            PS[e[0]] += 1
            for i in range(self.C - self.k):
                if level == 0 and np.count_nonzero(anti_PS==0) < self.k:
                    break
                anti_PS[e[-i - 1]] += 1
        #print(PS, anti_PS)
        # level 1
        if self.V in anti_PS:
            remove1 = np.where(anti_PS == self.V)[0]
            #print(remove1)
            # not_remove = np.where(PS != 0)[0]
            # remove1 = remove1[~np.isin(remove1, not_remove)]
            # print(remove1)
            Cand = np.delete(Cand, remove1)
            #print(len(remove1), 'удалится кандидатов на первом уровне')
            #print(Cand)
            self.candidates = np.delete(self.candidates, remove1, axis = 1)
            self.C = len(self.candidates[0])
            self.dist_matrix = np.delete(self.dist_matrix, remove1, axis = 0)
            self.add_matrices(self.dist_matrix)
            #self.make_matrix()
            self.VoteLists = np.argsort(self.dist_matrix, axis=0)
            PS = np.zeros(self.C)
            for e in self.VoteLists.T:
                PS[e[0]] += 1

        # level 2
        if level > 1:
            #print(PS)
            remove2 = np.where(PS == 0)[0]
            #print(len(remove2), 'удалится кандидатов на втором уровне')
            if len(Cand) - len(remove2) < self.k:
                if len(Cand) - self.k > 0:
                    sliceObj = slice(0, len(Cand) - self.k)
                    remove2 = remove2[sliceObj]
                else:
                    remove2 = np.array([])
            Cand = np.delete(Cand, remove2)
            #print(Cand)
            self.candidates = np.delete(self.candidates, remove2, axis = 1)
            self.C = len(self.candidates[0])
            self.dist_matrix = np.delete(self.dist_matrix, remove2, axis=0)
            self.add_matrices(self.dist_matrix)
            #self.make_matrix()
            self.VoteLists = np.argsort(self.dist_matrix, axis=0)
            #print('after:', self.dist_matrix)

    # def meanDist(self):
    #     Candists = np.ones(self.C)
    #     Candists = Candists*999999
    #     closestC = np.zeros(self.C)
    #     for i in range(self.C):
    #         for j in range(self.C):
    #             d = np.sqrt((self.candidates[0][i] - self.candidates[0][j])**2 + (self.candidates[1][i] - self.candidates[1][j])**2)
    #             if d < Candists[i] and i != j:
    #                 Candists[i] = d
    #                 closestC[i] = j
    #     self.mD = np.mean(Candists)
    #     return self.mD
    def ComDec(self):
        #Com = np.zeros((2, self.k))
        #j = 0
        self.committee_id = []
        for i in range(self.C):
            if self.decision[i] == 1:
                #Com[0][j] = self.candidates[0][i]
                #Com[1][j] = self.candidates[1][i]
                self.committee_id.append(i)
                #j += 1
        self.committee_id = np.array(self.committee_id)
        #print('ids: ',  self.committee_id)
        #self.committee = Com
    def Calc_Score_old(self):
        distances2 = np.zeros((self.k, self.V))
        for i in range(self.k):
            for j in range(self.V):
                distances2[i][j] = ((self.voters[0][j]-self.committee[0][i])**2 + (self.voters[1][j] - self.committee[1][i])**2)
                #print('1 scorebetween', i, 'and', j, ':', np.sqrt(distances2[i][j]))
        d2 = np.sort(distances2, axis=0)
        Score = np.zeros(self.V)
        for i in range(self.V):
            if self.sorted_dist_matrix[0][i] == d2[0][i] or d2[0][i] == 0:
                Score[i] = 1
            else:
                Score[i] = np.sqrt(self.sorted_dist_matrix[0][i]/d2[0][i])
        self.Score = np.mean(Score)
        return self.Score
    def Calc_Cost_old(self):
        self.Cost = 0
        distances = np.zeros((self.k, self.V))
        for i in range(self.k):
            for j in range(self.V):
                distances[i][j] = ((self.voters[0][j] - self.committee[0][i])**2 + (self.voters[1][j] - self.committee[1][i])**2)
                #print('1cost between', i, 'and', j, ':', np.sqrt(distances[i][j]))
        sorted_committee_matrix = np.sort(distances, axis=0)
        for i in range(self.V):
            self.Cost += np.sqrt(sorted_committee_matrix[0][i])

        return self.Cost
    def Calc_Score(self):
        #print('len', len(self.committee_id))
        distances2 = np.zeros((self.k, self.V))
        for i, id in enumerate(self.committee_id):
            for j in range(self.V):
                distances2[i][j] = self.dist_matrix[id][j]
                #print('2 score between', i, 'and', j, ':', np.sqrt(distances2[i][j]))
        d2 = np.sort(distances2, axis=0)
        Score = np.zeros(self.V)
        for i in range(self.V):
            if self.sorted_dist_matrix[0][i] == d2[0][i] or d2[0][i] == 0:
                Score[i] = 1
            else:
                Score[i] = np.sqrt(self.sorted_dist_matrix[0][i]/d2[0][i])
        self.Score = np.mean(Score)
        return self.Score
    def Calc_Cost(self):
        self.Cost = 0
        #print('len', len(self.committee_id))
        distances = np.zeros((self.k, self.V))
        for i, id in enumerate(self.committee_id):
            for j in range(self.V):
                distances[i][j] = self.dist_matrix[id][j]
                #print('2  cost between', i, 'and', j, ':', np.sqrt(distances[i][j]))
        sorted_committee_matrix = np.sort(distances, axis=0)
        for i in range(self.V):
            self.Cost += np.sqrt(sorted_committee_matrix[0][i])

        return self.Cost
    def draw(self, name = 'noname_pic'):
        fig, ax = plt.subplots()
        #figure(figsize=(16, 12), dpi=80)
        ax.scatter(self.voters[0], self.voters[1], s = 10, c = '#0101dd')
        ax.scatter(self.all_candidates[0], self.all_candidates[1], s = 30, c = '#fc171c', alpha = 0.75)
        ax.scatter(self.committee[0], self.committee[1], s = 70, c = '#01b700', alpha = 0.7)
        #plt.show()
        fig.set_size_inches(10, 10)
        fig.savefig('draw_election/' + name + '.png', dpi=100)
        plt.close(fig)
        #print("Score of " + name + " rule is ", self.Score)
        #print("median Score of STV rule is ", medianScore(Committee, Vamount, ComAmount, d1, V))

    # def BnB_rule(self, tol = 0.2, level = 2, depth = True, draw_name = 'BnB'):
    #     self.delete_max(level = level)
    #     decision = BnB(self.dist_matrix, self.k, tol = tol, depth = depth)
    #     self.decision = 1 - decision
    #     #print(self.decision)
    #     self.ComDec()
    #     #self.Calc_Score()
    #     #self.Calc_Cost()
    #
    #     #self.draw(name = draw_name)
    #     return self.Score
    def BnB_rule(self, tol = 0.2, level = 2, depth = True, draw_name = 'BnB'):
        self.delete_max(level = level)
        decision = BnB(self.dist_matrix, self.k, tol = tol, depth = depth, weights=self.weights)
        self.decision = 1 - decision
        #print(self.decision)
        self.ComDec()
        self.Calc_Score()
        #self.Calc_Cost()

        #self.draw(name = draw_name)
        #return self.Score
    # def SNTV_rule(self, draw_name = 'SNTV'):
    #     PS = np.zeros(self.C)
    #     for e in self.VoteLists.T:
    #         PS[e[0]] += 1
    #     top = np.argsort(-PS)
    #     dec = np.zeros(self.C)
    #     for cand in top[:self.k]:
    #         dec[cand] = 1
    #     self.decision = dec
    #     self.ComDec()
    #     #self.Calc_Score()
    #     #self.Calc_Cost()
    #
    #     #print('first: ', self.Score, self.Cost)
    #     #print('second: ', self.Calc_Score2(), self.Calc_Cost2())
    #     #self.draw(name =  draw_name)
    #     return self.Score
    def SNTV_rule(self, draw_name = 'SNTV'):
        PS = np.zeros(self.C)
        for voter, e in enumerate(self.VoteLists.T):
            PS[e[0]] += self.weights[voter]
        #print('PS', PS)
        top = np.argsort(-PS)
        dec = np.zeros(self.C)
        for cand in top[:self.k]:
            dec[cand] = 1
        self.decision = dec
        self.ComDec()
        #self.Calc_Score()
        #self.Calc_Cost()

        #print('Score: ', self.Score)
        #print('second: ', self.Calc_Score2(), self.Calc_Cost2())
        #self.draw(name =  draw_name)
        #return self.Score
    # def STV_rule_new(self, draw_name = 'STV'):
    #     elected = 0
    #     PS = np.zeros(self.C)
    #     for e in self.VoteLists.T:
    #         PS[e[0]] += 1
    #     n = self.V
    #     dec = np.zeros(self.C)
    #     while (elected < self.k):
    #         c = np.nanargmax(PS)
    #         q = (n + 1) // (self.k + 1) + 1
    #         if PS[c] >= q:
    #             dec[c] = 1
    #

    # def STV_rule(self, draw_name = 'STV'):
    #     elected = 0
    #     deleted = []
    #     Votes = self.VoteLists.T
    #     dec = np.zeros(self.C)
    #     flag = 0
    #     while (elected < self.k):
    #         n = len(Votes)
    #         #print('Votes', Votes)
    #         #print("size:", n)
    #         q = (n + 1)//(self.k + 1) + 1
    #         #print("q:", q)
    #         PS = np.zeros(self.C)
    #         if flag == 1:
    #             PS[deleted] = None
    #         for e in Votes:
    #             PS[e[0]] += 1
    #         #print(PS)
    #         c = np.nanargmax(PS)
    #         #print("c:", c, "PS:", PS[c])
    #         if PS[c] >= q:
    #             #print('больше')
    #             dec[c] = 1
    #             elected += 1
    #             ind = np.where(Votes == c)
    #             need = np.where(ind[1] == 0)
    #             #print(ind, need)
    #             for_remove = ind[0][need]
    #             #print('for_remove', for_remove)
    #             if len(for_remove) > q:
    #                 sdvig = np.random.randint(0, len(for_remove) - q)
    #                 rem = for_remove[(0 + sdvig): (q + sdvig)]
    #             else:
    #                 rem = for_remove
    #
    #             #print('rem', rem)
    #             Votes = np.delete(Votes, rem, 0)
    #             #print('Votes', Votes)
    #             ind = np.where(Votes == c)
    #             #print('ind', ind)
    #             Votes1 = []
    #             for i in range(len(Votes)):
    #                 Votes1.append(np.delete(Votes[i], ind[1][i]))
    #             Votes = np.array(Votes1)
    #             #print('Votes', Votes)
    #         else:
    #             #print('больше  нету')
    #             c = np.nanargmin(PS)
    #             #print("c else:", c)
    #             ind = np.where(Votes == c)
    #             #print(ind)
    #             Votes1 = []
    #             for i in range(len(Votes)):
    #                 Votes1.append(np.delete(Votes[i], ind[1][i]))
    #             Votes = np.array(Votes1)
    #             #print('Votes', Votes)
    #         deleted.append(c)
    #         flag = 1
    #         #print('deleted', deleted)
    #     #print(dec)
    #
    #     self.decision = dec
    #     self.ComDec()
    #     self.Calc_Score()
    #     self.Calc_Cost()
    #     #self.draw(name =  draw_name)
    #
    #     return self.Score
    def STV_star(self):
        deleted = []
        to_elect = self.k
        #print('COM SIZE', self.k)
        weights_sum = self.weights.sum()
        Votes = copy.deepcopy(self.VoteLists.T)
        STV_weights = copy.deepcopy(self.weights)
        dec = np.zeros(self.C)
        quota = weights_sum / (to_elect + 1)
        flag = 0
        while to_elect > 0:
            PS = np.zeros(self.C)
            if flag == 1:
                PS[deleted] = None
            for voter, e in enumerate(Votes):
                PS[e[0]] += STV_weights[voter]
            if len(PS[~np.isnan(PS)]) == to_elect:
                 for i in range(to_elect):
                    c = np.nanargmax(PS)
                    dec[c] = 1
                    PS[c] = None
                 break
            if len(Votes) < to_elect:
                PS = np.zeros(self.C)
                if flag == 1:
                    PS[deleted] = None
                for voter, e in enumerate(Votes):
                    for l in range(min(len(e), 10)):
                        PS[e[l]] += STV_weights[voter]/(2**l)
                for i in range(to_elect):
                    c = np.nanargmax(PS)
                    dec[c] = 1
                    PS[c] = None
                break
            #print('quota', quota)
            #print(to_elect)
            #print('V', len(Votes))
            #if len(PS[~np.isnan(PS)]) < 30:
            #    print('PS', PS[~np.isnan(PS)])

            c = np.nanargmax(PS)
            if PS[c] >= quota:
                dec[c] = 1
                # новые значения
                to_elect -= 1
                if to_elect == 0:
                    break
                #print(to_elect)
                ind = np.where(Votes == c)
                #print('ind', ind)
                need = np.where(ind[1] == 0)
                for_remove = ind[0][need]
                # удаление избирателей, у которых первый в списке
                Votes = np.delete(Votes, for_remove, 0)
                STV_weights = np.delete(STV_weights, for_remove)
                # удаление кандидата
                ind = np.where(Votes == c)
                # print('ind', ind)
                Votes1 = []
                for i in range(len(Votes)):
                    Votes1.append(np.delete(Votes[i], ind[1][i]))
                Votes = np.array(Votes1)

                quota = quota - (PS[c] - quota)/to_elect
                weights_sum -= PS[c]
            else:
                c = np.nanargmin(PS)
                ind = np.where(Votes == c)
                Votes1 = []
                for i in range(len(Votes)):
                    Votes1.append(np.delete(Votes[i], ind[1][i]))
                Votes = np.array(Votes1)
            deleted.append(c)
            flag = 1
        self.decision = dec
        self.ComDec()
        #self.Calc_Score()
        #self.Calc_Cost()
        # self.draw(name =  draw_name)
        #self.Calc_Score()
        #return self.Score
    def STV_basic(self):
        to_elect = self.k
        weights_sum = self.weights.sum()
        #print(weights_sum)
        Votes = copy.deepcopy(self.VoteLists.T)
        STV_weights = copy.deepcopy(self.weights)
        dec = np.zeros(self.C)
        quota = weights_sum / (to_elect + 1)
        #print('quota', quota)
        PS = np.zeros(self.C)
        for voter, e in enumerate(Votes):
            PS[e[0]] += STV_weights[voter]
        while to_elect > 0:
            if len(PS[~np.isnan(PS)]) == to_elect or len(Votes) < to_elect:
                 for i in range(to_elect):
                    c = np.nanargmax(PS)
                    dec[c] = 1
                    PS[c] = None
                 break


            #print(np.nansum(PS))
            #print(to_elect)
            c = np.nanargmax(PS)
            if PS[c] >= quota - 0.001:
                dec[c] = 1
                # новые значения
                to_elect -= 1
                if to_elect == 0:
                    break
                ind = np.where(Votes == c)
                need = np.where(ind[1] == 0)
                for_remove = ind[0][need]

                # перераспределение голосов среди вторых кандидатов
                for voter in for_remove:
                    PS[Votes[voter][1]] += STV_weights[voter] * (PS[c] - quota)/len(for_remove)
                    #print(STV_weights[voter],'*(', PS[c],'-',quota,')/',len(for_remove))
                    #print(voter, Votes[voter][1], PS[Votes[voter][1]])

                # удаление кандидата
                Votes1 = []
                for i in range(len(Votes)):
                    Votes1.append(np.delete(Votes[i], ind[1][i]))
                Votes = np.array(Votes1)

            else:
                c = np.nanargmin(PS)
                ind = np.where(Votes == c)
                need = np.where(ind[1] == 0)
                #print("need", need)
                for_remove = ind[0][need]
                #print("for remove", for_remove)
                # Votes = np.delete(Votes, for_remove, 0)
                debug_sum = 0
                for voter in for_remove:
                    PS[Votes[voter][1]] += STV_weights[voter]*(PS[c]/len(for_remove))
                    debug_sum += STV_weights[voter]
                    #print(voter, Votes[voter][1], PS[Votes[voter][1]])
                # if debug_sum > PS[c]:
                #     print(debug_sum, 'and', PS[c])
                #     print(for_remove)
                #     print(Votes[for_remove])
                #     print('PS', PS)
                #print(STV_weights)
                Votes1 = []
                for i in range(len(Votes)):
                    Votes1.append(np.delete(Votes[i], ind[1][i]))
                Votes = np.array(Votes1)
            #print('Votes', Votes)
            PS[c] = None

        self.decision = dec
        self.ComDec()
        #self.Calc_Score()
        #self.Calc_Cost()
        # self.draw(name =  draw_name)

        #return self.Score
    # def STV_basic_old(self, draw_name = 'STV'):
    #     elected = 0
    #     deleted = []
    #     Votes = self.VoteLists.T
    #     dec = np.zeros(self.C)
    #     flag = 0
    #     while (elected < self.k):
    #         n = len(Votes)
    #         #print('Votes', Votes)
    #         #print("size:", n)
    #         q = (n + 1)//(self.k + 1) + 1
    #         #print("q:", q)
    #         PS = np.zeros(self.C)
    #         if flag == 1:
    #             PS[deleted] = None
    #         for e in Votes:
    #             PS[e[0]] += 1
    #         #print(PS)
    #         c = np.nanargmax(PS)
    #         #print("c:", c, "PS:", PS[c])
    #         if PS[c] >= q:
    #             #print('больше')
    #             dec[c] = 1
    #             elected += 1
    #             ind = np.where(Votes == c)
    #             need = np.where(ind[1] == 0)
    #             #print(ind, need)
    #             for_remove = ind[0][need]
    #             #print('for_remove', for_remove)
    #             if len(for_remove) > q:
    #                 sdvig = np.random.randint(0, len(for_remove) - q)
    #                 rem = for_remove[(0 + sdvig): (q + sdvig)]
    #             else:
    #                 rem = for_remove
    #
    #             #print('rem', rem)
    #             Votes = np.delete(Votes, rem, 0)
    #             #print('Votes', Votes)
    #             #print('ind', ind)
    #             Votes1 = []
    #             for i in range(len(Votes)):
    #                 Votes1.append(np.delete(Votes[i], ind[1][i]))
    #             Votes = np.array(Votes1)
    #             #print('Votes', Votes)
    #         else:
    #             #print('больше  нету')
    #             c = np.nanargmin(PS)
    #             #print("c else:", c)
    #             ind = np.where(Votes == c)
    #             #print(ind)
    #             Votes1 = []
    #             for i in range(len(Votes)):
    #                 Votes1.append(np.delete(Votes[i], ind[1][i]))
    #             Votes = np.array(Votes1)
    #             #print('Votes', Votes)
    #         deleted.append(c)
    #         flag = 1
    #         #print('deleted', deleted)
    #     #print(dec)
    #
    #     self.decision = dec
    #     self.ComDec()
    #     self.Calc_Score()
    #     self.Calc_Cost()
    #     #self.draw(name =  draw_name)
    #
    #     return self.Score