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

#E_test = election(60, 35, 34, True)
#E_test.BnB_rule(level = 0, depth = True)

testing_BnB = Test(10, 40, 10)
testing_BnB.test_BnB_time()
print(testing_BnB.time_score_lists)