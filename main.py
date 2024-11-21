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

#E_test = election(30, 7, 3, True)
#E_test.STV_rule('huynya')
#E_test.BnB_rule(level = 0, depth = True)

testing_BnB = Test(50, 200, 25, commit_size = 9)
#testing_BnB.STV_test()
testing_BnB.BnB_level_V()
#testing_BnB.test_rules()
print('finish')
#print(testing_BnB.Scores)