import numpy as np
import gym
import numpy as np
import os
from gym import spaces
from stable_baselines3 import PPO
import math
from mosek.fusion import *
from itertools import combinations, product, chain


health_bins = 4
health_bin_edges = np.linspace(0, 1, health_bins + 1)
utility_bin_edges = np.array([0, 2, 3])

for health in [0, 0.1, 0.3, 0.6, 0.9, 1]:
    for utility in [0, 1, 2, 3]:
        utility_bin = int(np.digitize(utility, utility_bin_edges) - 1)
        health_bin = int(np.digitize(health, health_bin_edges) - 1)
        category = min(utility_bin * health_bins + health_bin, 
        print(f"health: {health}, utility: {utility}, category: {category}")