import numpy as np
import gym
import numpy as np
import os
from gym import spaces
from stable_baselines3 import PPO
import math
from mosek.fusion import *
from itertools import combinations, product, chain

utility_bins = 4
utility_bin_edges = np.array([0, 2, 3])
for utility in range(4):
    print(np.digitize(utility, utility_bin_edges) - 1)

x = set()
x.add(1)
y = set(x)
y.add(2)
y.remove(1)
print(x, y)