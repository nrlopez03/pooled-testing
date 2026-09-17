# %% [markdown]
# ## Requirements

# %%
import pandas as pd
import numpy as np
import random
import math
import itertools
import ast
import time
from math import prod
from scipy.stats import bernoulli
from itertools import combinations
from itertools import combinations_with_replacement
from itertools import product
from itertools import chain
from collections import defaultdict
import tensorflow as tf
import logging
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from tqdm.notebook import tqdm
import seaborn as sns
from sklearn.linear_model import LinearRegression
import os
import torch
from joblib import Parallel, delayed
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import csv
import ast
from mosek.fusion import *
import math
from scipy.optimize import brentq
from math import log, exp
import gurobipy as gp
from gurobipy import GRB
from collections import Counter
import multiprocessing

# Total number of CPUs available on the system
print("Total CPUs available:", os.cpu_count())

# CPUs available in the SLURM allocation
print("CPUs allocated by SLURM:", multiprocessing.cpu_count())

n_jobs = multiprocessing.cpu_count()
print(f"n_jobs: {n_jobs}")

plt.rcParams['figure.dpi'] = 300

# %%
# seed = 12

# # Standard Python random module
# random.seed(seed)

# # NumPy
# np.random.seed(seed)

# # PyTorch (for both CPU and GPU)
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)

# # TensorFlow
# tf.random.set_seed(seed)

# %% [markdown]
# ## Model

# %%
# define parameters

B = 2
k = 2
N = 3
G = N

# %%
# list of (id, utility, probability of healthy) tuples

def create_agents(N = N, uIntegers=False):
  agents = list()

  for i in range(N):
    if uIntegers:
      agents.append((i, round(random.random() * 100) + 1, random.random()))
    else:
      agents.append((i, random.random(), random.random()))
  return agents

agents = create_agents()

# print(agents)

# %%
# print results of a static solver

def printStatic(strategy, utility):
  print(f"\nStatic Strategy, Utility {utility}:")
  i = 1
  for group in strategy:
    if group:
      print(f"Group {i}: {group}")
      i += 1

# %%
# print results of a dynamic solver

def printDynamic(strategy, utility = 0, i = 1, greedy = False):
  if utility > 0:
    if greedy:
      print(f"\nGreedy Dynamic Strategy, Utility {utility}:")
    else:
      print(f"\nDynamic Strategy, Utility {utility}:")
  if len(strategy)>0:
    print(f"Test {i}: {strategy[0]}")
    i+=1
    if len(strategy[1]) > 0:
      print(('\t' * (i - 1)) + f"If positive, ", end="")
      printDynamic(strategy[1], 0, i)
    if len(strategy[2]) > 0:
        # i += 1
        print(('\t' * (i - 1)) + f"If negative, ", end="")
        printDynamic(strategy[2], 0, i)

# %%
# print results of a dynamic solver

def printTree(strategy, utility = 0, i = 1, greedy = False):
  if utility > 0:
    if greedy:
      print(f"\nGreedy Dynamic Strategy, Utility {utility}:")
    else:
      print(f"\nDynamic Strategy, Utility {utility}:")
  if strategy:
    print(f"Test {i}: {strategy[0]}")
    if strategy[1]:
      i+=1
      print(('\t' * (i - 1)) + f"If positive, ", end="")
      printTree(strategy[1], 0, i)
    if strategy[2]:
        # i += 1
        print(('\t' * (i - 1)) + f"If negative, ", end="")
        printTree(strategy[2], 0, i)

# %%
def generate_binary_tree(strategy):
    # Base case: if the strategy is empty, return None
    if not strategy:
        return None

    # Extract the current test and its positive and negative branches
    current_test = set([person[0] for person in strategy[0]])
    positive_branch = generate_binary_tree(strategy[1])
    negative_branch = generate_binary_tree(strategy[2])

    # Return a tuple representing the binary tree node
    return (current_test, positive_branch, negative_branch)

# %%
def tree_to_boolean_list(tree, N, bool_list=None):

    if bool_list is None:
        bool_list = []

    new_list = [False] * N

    # Unpack the tree node
    values, left, right = tree

    # Update the boolean list for this level
    for val in values:
        new_list[val] = True

    bool_list += new_list

    # Recurse for the left and right subtrees, moving to the next level
    if left:
        bool_list = tree_to_boolean_list(left, N, bool_list)
    if right:
        bool_list = tree_to_boolean_list(right, N, bool_list)

    return bool_list

# # Example usage:
# tree = ({0, 1}, ({1}, ({1}, None, None), ({1}, None, None)), ({2}, ({1}, None, None), ({1}, None, None)))

# boolean_list = tree_to_boolean_list(tree, N)


# %%
def boolean_list_to_tree(bool_list, N):
    # print(f"Input bool_list: {bool_list}")
    # print(f"Expected N: {N}, Actual Length: {len(bool_list)}")
    # assert len(bool_list) >= N, f"bool_list {bool_list} is shorter than expected {N}!"
    # current_values = set()
    # for i in range(N):
    #     print(f"Index: {i}, Value: {bool_list[i]}")
    #     if bool_list[i]:
    #         current_values.add(i)


    # Extract the current node values
    current_values = {i for i in range(N) if bool_list[i]}

    if len(bool_list) == N:
        return (current_values, None, None)

    # Move the index forward by N for the next segment
    cutoff = N + ((len(bool_list) - N) // 2)

    leftList = bool_list[N : cutoff]
    rightList = bool_list[cutoff:]

    # Recursively construct the left subtree
    left_subtree = boolean_list_to_tree(leftList, N)

    # Recursively construct the right subtree
    right_subtree = boolean_list_to_tree(rightList, N)

    return (current_values, left_subtree, right_subtree)

# # Example usage:
# boolean_list = [True, True, False, False, True, False, False, True, False, False, True, False, False, False, True, False, True, False, False, True, False]
# N = 3
# reconstructed_tree = boolean_list_to_tree(boolean_list, N)


# %% [markdown]
# ## Algorithms

# %% [markdown]
# ### Maximum Utility

# %%
def maxUtil(agents):
  util = 0
  for agent in agents:
    util += agent[1] * agent[2]
  return util

# %%
# maxUtil(agents)

# %% [markdown]
# ### Static Non-Pooled

# %%
def solveStaticNoPool(agents, B=B):
  utils = []
  for agent in agents:
    utils.append(agent[1] * agent[2])
  utils = sorted(utils, reverse=True)
    # Take the first B elements and sum them
  util = sum(utils[:B])
  return util

# %% [markdown]
# ### Static Non-Overlapping

# %% [markdown]
# Generate and calculate utility of each partition

# %%
def solveStaticNonOverlap(agents, G = G, B = B):

  def generate_non_overlapping_subsets(agents, G, B):
    # Generate all possible subsets of agents with sizes up to G
    subsets = []
    for r in range(1, G+1):
        subsets.extend(combinations(agents, r))

    # Generate combinations of B non-overlapping subsets
    combos = combinations(subsets, B)

    # Filter combinations to keep only non-overlapping ones
    non_overlapping_combinations = []
    for combination in combos:
        combined_set = set()
        is_non_overlapping = True
        for subset in combination:
            if combined_set.intersection(subset):
                is_non_overlapping = False
                break
            combined_set.update(subset)
        if is_non_overlapping:
            non_overlapping_combinations.append(combination)

    return non_overlapping_combinations

  def strategyUtilitySNO(combination):
    utility = 0
    for group in combination:
      groupUtility = 0
      groupHealthy = 1
      for person in group:
        groupUtility += person[1]
        groupHealthy *= person[2]
      utility += groupHealthy * groupUtility
    return utility

  strategy = dict()
  combos = generate_non_overlapping_subsets(agents, G, B)
  for combination in combos:
    strategy[combination] = strategyUtilitySNO(combination)
  optimal = max(strategy, key=strategy.get)
  return optimal, strategy[optimal]

# %%
# strategy, utility = solveStaticNonOverlap(agents)
# printStatic(strategy, utility)

# %% [markdown]
# ### Determining Updated Priors, Utilities

# %% [markdown]
# #### Bayes' Theorem

# %%
def bayesTheorem(agents, posGroups, negAgents):

  posGroups = list(posGroups)
  posGroups = sorted(posGroups, key=len)

  for posGroup in posGroups:
    for posGroupCompare in posGroups:
      if posGroup.issubset(posGroupCompare) and posGroup != posGroupCompare:
        posGroups.remove(posGroupCompare)

  newAgents = []

  for agent in agents:
    if agent[0] in set(negAgents):
      newAgents.append((agent[0], 0, 1))
    else:
      newAgents.append(agent)

  agentDict = {}
  for agent in newAgents:
    agentDict[agent[0]] = (agent[1], agent[2])


  def all_combinations(frozen_sets):

    def pick_at_least_one(agents_set):
      # Generate combinations of different sizes from the set
      combinations_list = []
      for r in range(1, len(agents_set) + 1):
          combinations_list.extend(combinations(agents_set, r))
      return combinations_list

    possibilities = [pick_at_least_one(group) for group in frozen_sets]
    product_result = product(*possibilities)

    # Combine the elements of each tuple into a single set
    combos = {frozenset(chain(*comb)) for comb in product_result}

    return combos

  def getProb(groups):
    involvedAgents = frozenset(chain(*groups))

    possiblePos = all_combinations(groups)

    prob = 0
    for possibility in possiblePos:
      currentOutcome = 1
      for id in involvedAgents:
        if id in possibility:
          currentOutcome *= 1 - agentDict[id][1]
        else:
          currentOutcome *= agentDict[id][1]
      prob += currentOutcome

    return prob

  probPos = {}

  probPos[tuple(posGroups)] = getProb(posGroups)
  probPos[()] = 1

  finalAgents = []

  for agent in newAgents:

    # P(agent is Pos | posGroups) = P(posGroups | agent is Pos) * P(agent in Pos) / P(posGroups)
    # = P(pos Groups agent not in) * P(agent is Pos) / P(posGroups)

    if agent[2] == 1 or agent[2] == 0:
      finalAgents.append(agent)
      # agentDict[agent[0]] = (agent[1], agent[2])

    else:

      posNotIn = [posGroup for posGroup in posGroups if agent[0] not in posGroup]

      if tuple(posNotIn) not in probPos:

        probPos[tuple(posNotIn)] = getProb(posNotIn)

      pAgentPos = probPos[tuple(posNotIn)] * (1 - agent[2]) / probPos[tuple(posGroups)]
      health = 1 - pAgentPos

      finalAgents.append((agent[0], agent[1], health))
      # agentDict[agent[0]] = (agent[1], health)

  for agent in finalAgents:
    agentDict[agent[0]] = (agent[1], agent[2])

  return agentDict


"""
MICOP model for a single test that can be solved exactly with MOSEK.

Note that the objective value of this model is the log of the overall welfare!
"""
def solveConicSingle(agents, G=G, verbose=False):

    agents = [agent for agent in agents if agent[1] != 0 and agent[2] != 0]

    if len(agents) == 0:
        return tuple(), 0

    u = [agent[1] for agent in agents]
    q = [agent[2] for agent in agents]

    # Compute population size
    n = len(u)

    assert n == len(q), "Input vectors have different lengths."
    assert all(u[i] >= 0 for i in range(n)), "Utilities must be non-negative."
    assert all(q[i] >= 0 and q[i] <= 1 for i in range(n)), "Probabilities must lie between 0 and 1."

    # Hack alert: Conic program doesn't like -math.inf
    q = [(math.log(qi) if qi!= 0 else -100000000000) for qi in q]

    with Model('ceo1') as M:
        # Define variables
        x = M.variable("x", n, Domain.binary())
        y = M.variable("y", 1, Domain.unbounded())
        z = M.variable("z", 1, Domain.unbounded())
        d = M.variable("d", 1, Domain.equalsTo(1))
        t = Var.vstack(z.index(0), d.index(0), y.index(0))

        # Add objective
        M.objective("obj", ObjectiveSense.Maximize, Expr.add(y.index(0), Expr.dot(x, q)))

        # Add constraints
        M.constraint("ev", Expr.sub(Expr.dot(u, x), z.index(0)), Domain.equalsTo(0))
        M.constraint("expc", t, Domain.inPExpCone())

        # Pooled testing size constraint

        M.constraint("pool", Expr.sum(x), Domain.lessThan(G))

        M.solve()

        # print(agents)
        # print(tuple(agent for agent, level in zip(agents, np.round_(x.level())) if level == 1))
        # print(M.primalObjValue())

        utility = math.exp(M.primalObjValue())
        strategy = tuple(agent for agent, level in zip(agents, np.round_(x.level())) if level == 1)

        return strategy, utility
    
# test = create_agents(5)
# print(solveConicSingle(test, G=5))


# #### Gibbs MCMC Window Sampling

# %%
def GibbsMCMCWindow(agents, posGroups, negAgents, max_iterations=1000, tolerance=0.05, min_burn_in=50, window_size=100, confidence_level=0.95, n_bootstrap=1000):

    # Initialize agents with binary states (0: healthy, 1: infected), ensuring negAgents are always healthy
    def initialize_agents_binary(agents, negAgents):
        return {agent[0]: 0 if agent[0] in negAgents else np.random.choice([0, 1], p=[agent[2], 1-agent[2]]) for agent in agents}

    # Gibbs-like sampling: Update one agent's state given the rest of the current states
    def update_agent_state(agent, agent_to_groups, health_states, agent_prob, negAgents):
        """Update the state of the agent (0: healthy, 1: infected) based on the group constraints."""
        if agent in negAgents:
            return 0  # If the agent is in negAgents, they must always be healthy

        relevant_groups = agent_to_groups[agent]
        must_be_infected = False
        for group in relevant_groups:
            if all(health_states[other] == 0 for other in group if other != agent):
                must_be_infected = True
                break  # Once the agent must be infected, no need to check other groups

        if must_be_infected:
            return 1  # The agent must be infected to satisfy the group constraint
        else:
            return bernoulli.rvs(1 - agent_prob)
        
    unknownAgentIDs = {agent_id for posGroup in posGroups for agent_id in posGroup}

    unknownAgents = [agent for agent in agents if agent[0] in unknownAgentIDs]

    # Initialize binary health states
    health_states = initialize_agents_binary(unknownAgents, negAgents)

    # Precompute agent-to-groups mapping
    agent_to_groups = {agent[0]: [group for group in posGroups if agent[0] in group] for agent in unknownAgents}

    # Store history of probabilities to monitor convergence
    health_history = {agent[0]: [] for agent in unknownAgents}

    # Store counts of healthy samples for each agent
    healthy_counts = {agent[0]: 0 for agent in unknownAgents}

    # Run MCMC iterations
    for iteration in range(max_iterations):
        for agent in unknownAgents:
            # Update the state of each agent given the current states, except for agents in negAgents
            health_states[agent[0]] = update_agent_state(agent[0], agent_to_groups, health_states, agent[2], negAgents)

        # Record the number of times each agent is healthy after burn-in
        for agent in unknownAgents:
            health_history[agent[0]].append(health_states[agent[0]])
            if len(health_history[agent[0]]) > window_size:
                health_history[agent[0]].pop(0)

        # Check for convergence by comparing moving averages
        if iteration > min_burn_in:
            converged = True
            for agent in unknownAgents:
                if len(health_history[agent[0]]) >= window_size:
                    avg_health = np.mean(health_history[agent[0]])
                    prev_avg_health = np.mean(health_history[agent[0]][-window_size//2:])
                    if abs(avg_health - prev_avg_health) > tolerance:
                        converged = False
                        break
            if converged:
                burn_in = iteration
                break
    else:
        burn_in = max_iterations  # No convergence detected, use max iterations

    # After burn-in, continue sampling to calculate probabilities and bootstrap for confidence intervals
    post_burn_in_health_states = {agent[0]: [] for agent in unknownAgents}
    total_samples = max(max_iterations - burn_in, burn_in)
    for iteration in range(total_samples):
        for agent in unknownAgents:
            health_states[agent[0]] = update_agent_state(agent[0], agent_to_groups, health_states, agent[2], negAgents)

        # Record state for calculating probabilities and confidence intervals
        for agent in unknownAgents:
            if health_states[agent[0]] == 0:
                healthy_counts[agent[0]] += 1
            post_burn_in_health_states[agent[0]].append(health_states[agent[0]])

    # Calculate the final probabilities using the original count method
    final_probs = {agent[0]: healthy_counts[agent[0]] / total_samples for agent in unknownAgents}

    # Calculate bootstrap confidence intervals using post-burn-in health samples
    ci_intervals = {}
    for agent in agents:
        agent_id = agent[0]
        if agent_id in negAgents:
            ci_intervals[agent_id] = (1.0, 1.0)  # Always healthy
        elif agent_id in unknownAgentIDs:
            # Bootstrap to calculate confidence intervals for P(healthy) = 1 - P(infected)
            health_samples = np.array(post_burn_in_health_states[agent_id])
            prob_samples = [1 - np.mean(np.random.choice(health_samples, size=len(health_samples), replace=True)) for _ in range(n_bootstrap)]
            lower_bound = np.percentile(prob_samples, (1 - confidence_level) / 2 * 100)
            upper_bound = np.percentile(prob_samples, (1 + confidence_level) / 2 * 100)
            ci_intervals[agent_id] = (lower_bound, upper_bound)
        else: 
            ci_intervals[agent_id] = (agent[2], agent[2])

    # Prepare the agentDict to return, setting utility to 0 for negAgents
    agentDict = {}
    for agent in agents:
        agent_id = agent[0]
        if agent_id in negAgents:
            agentDict[agent_id] = (0, 1, (1.0, 1.0))  # Utility is 0, always healthy, confidence interval at (1.0, 1.0)
        elif agent_id in unknownAgentIDs:
            agentDict[agent_id] = (agent[1], final_probs[agent_id], ci_intervals[agent_id])  # Utility remains the same, update health probability and confidence interval
        else:
            agentDict[agent_id] = (agent[1], agent[2], ci_intervals[agent_id])

    return agentDict

# %% [markdown]
# #### Gibbs MCMC Sampling

# %%
def GibbsMCMC(agents, posGroups, negAgents, iterations=1000, burn_in=250):

    # Initialize agents with binary states (0: healthy, 1: infected), ensuring negAgents are always healthy
    def initialize_agents_binary(agents, negAgents):
        return {agent[0]: 0 if agent[0] in negAgents else np.random.choice([0, 1], p=[agent[2], 1-agent[2]]) for agent in agents}

    # Gibbs-like sampling: Update one agent's state given the rest of the current states
    def update_agent_state(agent, agent_to_groups, health_states, agent_prob, negAgents):
        """Update the state of the agent (0: healthy, 1: infected) based on the group constraints."""
        if agent in negAgents:
            return 0  # If the agent is in negAgents, they must always be healthy

        relevant_groups = agent_to_groups[agent]

        # Compute the probability that the agent should be infected based on their group constraints
        must_be_infected = False
        for group in relevant_groups:
            if all(health_states[other] == 0 for other in group if other != agent):
                must_be_infected = True
                break  # Once the agent must be infected, no need to check other groups

        if must_be_infected:
            return 1  # The agent must be infected to satisfy the group constraint
        else:
            # Otherwise, use their probability of being healthy (agent_prob)
            return bernoulli.rvs(1 - agent_prob)

    """Perform MCMC to update health states (binary: 0 or 1) and calculate probabilities."""
    # Initialize binary health states (0: healthy, 1: infected), ensuring negAgents are always healthy
    health_states = initialize_agents_binary(agents, negAgents)

    # Precompute agent-to-groups mapping to avoid recalculating during iterations
    agent_to_groups = {agent[0]: [group for group in posGroups if agent[0] in group] for agent in agents}

    # Store counts of healthy samples for each agent
    healthy_counts = {agent[0]: 0 for agent in agents}

    # Run MCMC iterations
    for iteration in range(iterations):
        for agent in agents:
            # Update the state of each agent given the current states, except for agents in negAgents
            health_states[agent[0]] = update_agent_state(agent[0], agent_to_groups, health_states, agent[2], negAgents)

        # After burn-in, record the number of times each agent is healthy
        if iteration >= burn_in:
            for agent in agents:
                if health_states[agent[0]] == 0:
                    healthy_counts[agent[0]] += 1

    # Calculate the final probabilities of being healthy
    total_samples = iterations - burn_in
    final_probs = {agent[0]: healthy_counts[agent[0]] / total_samples for agent in agents}

    # Prepare the agentDict to return, setting utility to 0 for negAgents
    agentDict = {}
    for agent in agents:
        agent_id = agent[0]
        if agent_id in negAgents:
            agentDict[agent_id] = (0, 1)  # Utility is 0, always healthy
        else:
            agentDict[agent_id] = (agent[1], final_probs[agent_id])  # Utility remains the same, update health probability

    return agentDict

# %% [markdown]
# #### Tests with Implementation

# %% [markdown]
# Printing Results

# %%
def printBayes(agentDict):

  print("Agent, P(Infected), Utility of Testing Negative")

  for id in agentDict:
    print(f"Agent {id}: {1 - agentDict[id][1]}, {agentDict[id][0]}")

# %%
def evaluateCI(gibbsAgentDict, bayesAgentDict):
    for id in gibbsAgentDict:
        if gibbsAgentDict[id][2][0] > bayesAgentDict[id][1] or gibbsAgentDict[id][2][1] < bayesAgentDict[id][1]:
            print(f"ID: {id}, Gibbs: {gibbsAgentDict[id][1]}, Bayes: {bayesAgentDict[id][1]}, does not fit {gibbsAgentDict[id][2]}")
    

# # %% [markdown]
# # Custom Testing

# # %%
# agents = [(0, 1, 1/2), (1, 1, 1/2), (2, 1, 1/2)]
# posGroups = [frozenset({0,1}), frozenset({1,2})]
# negAgents = {}
# # printBayes(bayesTheorem(agents, posGroups, negAgents))
# # print("\n")
# # printBayes(GibbsMCMCWindow(agents, posGroups, negAgents))
# # print("\n")
# evaluateCI(GibbsMCMCWindow(agents, posGroups, negAgents), bayesTheorem(agents, posGroups, negAgents))

# # %% [markdown]
# # Testing Overlap Results

# # %% [markdown]

# # %%
# agents = [(i + 1, 1/2, 1/2) for i in range(9)]
# posGroups = [frozenset({7, 3, 1, 4}), frozenset({2, 1, 5, 6})]
# negAgents = {}
# # printBayes(bayesTheorem(agents, posGroups, negAgents))
# # print("\n")
# # printBayes(GibbsMCMCWindow(agents, posGroups, negAgents))
# GibbsMCMCWindow(agents, posGroups, negAgents)
# # print("\n")
# evaluateCI(GibbsMCMCWindow(agents, posGroups, negAgents), bayesTheorem(agents, posGroups, negAgents))

# # %%
# agents = [(i + 1, 1/2, 1/2) for i in range(9)]
# posGroups = [frozenset({7, 3, 1, 4}), frozenset({2, 1, 5, 6}), frozenset({1})]
# negAgents = {}
# # printBayes(bayesTheorem(agents, posGroups, negAgents))
# # print("\n")
# # printBayes(GibbsMCMCWindow(agents, posGroups, negAgents))
# # print("\n")
# evaluateCI(GibbsMCMCWindow(agents, posGroups, negAgents), bayesTheorem(agents, posGroups, negAgents))

# # %%
# agents = [(i + 1, 1/2, 1/2) for i in range(9)]
# posGroups = [frozenset({7, 3, 1, 4}), frozenset({2, 1, 5, 6}), frozenset({1}), frozenset({8, 6}), frozenset({9, 7})]
# negAgents = {}
# # printBayes(bayesTheorem(agents, posGroups, negAgents))
# # print("\n")
# # printBayes(GibbsMCMC(agents, posGroups, negAgents))
# # print("\n")
# evaluateCI(GibbsMCMCWindow(agents, posGroups, negAgents), bayesTheorem(agents, posGroups, negAgents))

# # %%
# agents = [(i + 1, 1/2, 1/2) for i in range(9)]
# posGroups = [frozenset({1}), frozenset({8, 6}), frozenset({9, 7})]
# negAgents = {}
# # printBayes(bayesTheorem(agents, posGroups, negAgents))
# # print("\n")
# # printBayes(GibbsMCMCWindow(agents, posGroups, negAgents))
# # print("\n")
# evaluateCI(GibbsMCMCWindow(agents, posGroups, negAgents), bayesTheorem(agents, posGroups, negAgents))

# %% [markdown]
# ### Static Overlapping

# %%
def solveStaticOverlap(agents, G = G, B = B):

  def generate_overlapping_subsets(agents, G, B):

    # Generate all possible subsets of agents with sizes up to G
    subsets = []
    for r in range(1, G+1):
        subsets.extend(combinations(agents, r))

    # Generate combinations of B subsets without enforcing overlap
    combos = combinations(subsets, B)
    return combos

  def groupHelp(group, posGroups = frozenset(), negAgents = frozenset()):

    groupIDs = frozenset({person[0] for person in group})
    groupUtility = 0
    groupHealthy = 1
    for posGroup in posGroups:
      if posGroup.issubset(groupIDs):
        groupHealthy = 0

    agentDict = bayesTheorem(agents, posGroups, negAgents)

    for person in group:
      groupUtility += agentDict[person[0]][0]
      groupHealthy *= agentDict[person[0]][1]

    groupUtility *= groupHealthy
    return groupHealthy, groupUtility


  def strategyUtility(combination, posGroups = frozenset(), negAgents = frozenset()):

    utility = 0

    if len(combination) > 0:

      firstGroup = combination[0]
      groupIDs = frozenset({person[0] for person in firstGroup})

      firstHealthy, firstUtility = groupHelp(firstGroup, posGroups, negAgents)
      utility += firstUtility

      # positive test
      if firstHealthy < 1:
        posScenario = combination[1:]

        newPosGroups = set(posGroups.copy())

        for posGroup in newPosGroups:
          if groupIDs.issubset(posGroup):
            newPosGroups.remove(posGroup)

        newPosGroups.add(groupIDs.difference(negAgents))

        utility += strategyUtility(posScenario, frozenset(newPosGroups), frozenset(negAgents)) * (1-firstHealthy)

      # negative test
      if firstHealthy > 0:

        negScenario = combination[1:]

        newPosGroups = set()
        for posGroup in posGroups:
          newPosGroups.add(posGroup.difference(groupIDs))

        newNegAgents = set(negAgents.copy())
        newNegAgents.update(groupIDs)

        utility += strategyUtility(negScenario, frozenset(newPosGroups), frozenset(newNegAgents)) * firstHealthy

    return utility

  strategy = dict()
  combos = generate_overlapping_subsets(agents, G, B)
  for combination in combos:
    strategy[combination] = strategyUtility(combination)
  optimal = max(strategy, key=strategy.get)
  return optimal, strategy[optimal]

# # %%
# agents = [(0, 1, 0.5), (1, 1, 0.5), (2, 1, 1)]
# staticStrategy, staticUtility = solveStaticOverlap(agents)
# # printStatic(staticStrategy, staticUtility)

# # %%
# # print(staticStrategy)

# # %%
# agents = [(0, 0.022080006061812485, 0.8205655186482051), (1, 0.760970132052074, 0.5930101907614745), (2, 0.42354957403482274, 0.6984521403961065), (3, 0.006600413727435472, 0.2859953353854493), (4, 0.616360236705176, 0.29752798376295153)]
# staticStrategy, staticUtility = solveStaticOverlap(agents, B= 4)
# printStatic(staticStrategy, staticUtility)

# %% [markdown]
# ### Dynamic

# %%
def solveDynamic(agents, G = G, B = B, posGroups = frozenset(), negAgents = frozenset()):

  if B == 0:
    return [], 0

  def generate_subsets(agents, G=G):
      subsets = []
      for r in range(1, G + 1):
          subsets.extend(combinations(agents, r))
      return subsets

  strategy = dict()
  combos = generate_subsets(agents, G)
  for combination in combos:
    utility = 0

    # first test
    firstTest = combination
    firstUtility = 0
    firstHealthy = 1

    # remaining agents
    remaining = [person for person in agents if person not in firstTest]

    # utility, P(Healthy) of first test
    firstIDs = frozenset({person[0] for person in firstTest})
    if posGroups:
      for posGroup in posGroups:
        if posGroup.issubset(firstIDs):
          firstHealthy = 0
    else:
      posGroups = frozenset()

    agentDict = bayesTheorem(agents, posGroups, negAgents)

    for person in firstTest:
      firstUtility += agentDict[person[0]][0]
      firstHealthy *= agentDict[person[0]][1]
    utility += firstUtility * firstHealthy

    # positive scenario
    if firstHealthy < 1:

      newPosGroups = set(posGroups.copy())
      newPosGroups.add(firstIDs.difference(negAgents))

      posStrategy, posUtility = solveDynamic(agents, G, B-1, frozenset(newPosGroups), frozenset(negAgents))

      utility += (1- firstHealthy) * posUtility

    else:

      posStrategy = []

    # negative scenario

    if remaining and firstHealthy > 0:
      newPosGroups = set()
      for posGroup in posGroups:
        newPosGroups.add(posGroup.difference(firstIDs))

      newNegAgents = set(negAgents.copy())
      newNegAgents.update(firstIDs)


      negStrategy, negUtility = solveDynamic(remaining, G, B-1, frozenset(newPosGroups), frozenset(newNegAgents))
      utility += firstHealthy * negUtility

    else:

      negStrategy = []

    strategy[(tuple(firstTest), tuple(posStrategy), tuple(negStrategy))] = utility

  optimal = max(strategy, key=strategy.get)
  return optimal, strategy[optimal]

# # %%
# agents = [(0, 1, 0.5), (1, 1, 0.5), (2, 1, 1)]
# dynamicStrategy, dynamicUtility = solveDynamic(agents)

# # printDynamic(dynamicStrategy, dynamicUtility)

# # %% [markdown]
# # ### Dynamic Strategy Analysis and Representation

# # %%
# agents = [(0, 0.129, 0.5562), (1, 0.17483, 1), (2, 0.569, 0.12)]

# dynamicStrategy, dynamicUtility = solveDynamic(agents)

# printDynamic(dynamicStrategy, dynamicUtility)

# print(generate_binary_tree(dynamicStrategy))

# print(tree_to_boolean_list(generate_binary_tree(dynamicStrategy), N))

# print(boolean_list_to_tree(tree_to_boolean_list(generate_binary_tree(dynamicStrategy), N), N))

# %%
def analyzeTree(tree, agents, posGroups = frozenset(), negAgents = frozenset()):

  utility = 0

  # first test
  firstTest, posStrategy, negStrategy = tree
  firstUtility = 0
  firstHealthy = 1

  # remaining agents
  remaining = [person for person in agents if person not in firstTest]

  # utility, P(Healthy) of first test
  firstIDs = frozenset(firstTest)
  if posGroups:
    for posGroup in posGroups:
      if posGroup.issubset(firstIDs):
        firstHealthy = 0
  else:
    posGroups = frozenset()

  agentDict = bayesTheorem(agents, posGroups, negAgents)

  for person in firstTest:
    firstUtility += agentDict[person][0]
    firstHealthy *= agentDict[person][1]
  utility += firstUtility * firstHealthy

  # positive scenario
  if firstHealthy < 1:

    newPosGroups = set(posGroups.copy())
    newPosGroups.add(firstIDs.difference(negAgents))

    posUtility = analyzeTree(posStrategy, agents, frozenset(newPosGroups), frozenset(negAgents)) if posStrategy else 0

    utility += (1- firstHealthy) * posUtility

  else:

    posStrategy = []

  # negative scenario

  if remaining and firstHealthy > 0:
    newPosGroups = set()
    for posGroup in posGroups:
      newPosGroups.add(posGroup.difference(firstIDs))

    newNegAgents = set(negAgents.copy())
    newNegAgents.update(firstIDs)


    negUtility = analyzeTree(negStrategy, agents, frozenset(newPosGroups), frozenset(newNegAgents)) if negStrategy else 0
    utility += firstHealthy * negUtility

  else:

    negStrategy = []

  return utility

# %%
def analyzeTreeGibbs(tree, agents, posGroups = frozenset(), negAgents = frozenset(), confidence_level=0.95):

  utility = 0
  lowUtility = 0
  highUtility = 0

  # first test
  firstTest, posStrategy, negStrategy = tree
  firstUtility = 0
  firstHealthy = 1
  lowFirstHealthy = 1
  highFirstHealthy = 1

  # remaining agents
  remaining = [person for person in agents if person not in firstTest]

  # utility, P(Healthy) of first test
  firstIDs = frozenset(firstTest)
  if posGroups:
    for posGroup in posGroups:
      if posGroup.issubset(firstIDs):
        firstHealthy = 0
  else:
    posGroups = frozenset()

  agentDict = GibbsMCMCWindow(agents, posGroups, negAgents, confidence_level=confidence_level) 

  for person in firstTest:
    firstUtility += agentDict[person][0]
    firstHealthy *= agentDict[person][1]
    lowFirstHealthy *= agentDict[person][2][0]
    highFirstHealthy *= agentDict[person][2][1]
  utility += firstUtility * firstHealthy
  lowUtility += firstUtility * lowFirstHealthy
  highUtility += firstUtility * highFirstHealthy

  # positive scenario
  if firstHealthy < 1:

    newPosGroups = set(posGroups.copy())
    newPosGroups.add(firstIDs.difference(negAgents))

    posUtility, (lowPosUtility, highPosUtility) = analyzeTreeGibbs(posStrategy, agents, frozenset(newPosGroups), frozenset(negAgents), confidence_level=confidence_level) if posStrategy else (0, (0, 0))

    utility += (1- firstHealthy) * posUtility
    lowUtility += (1 - lowFirstHealthy) * lowPosUtility
    highUtility += (1 - highFirstHealthy) * highPosUtility

  # negative scenario

  if remaining and firstHealthy > 0:
    newPosGroups = set()
    for posGroup in posGroups:
      newPosGroups.add(posGroup.difference(firstIDs))

    newNegAgents = set(negAgents.copy())
    newNegAgents.update(firstIDs)

    negUtility, (lowNegUtility, highNegUtility) = analyzeTreeGibbs(negStrategy, agents, frozenset(newPosGroups), frozenset(newNegAgents), confidence_level=confidence_level) if negStrategy else (0, (0, 0))

    utility += firstHealthy * negUtility
    lowUtility += lowFirstHealthy * lowNegUtility
    highUtility += highFirstHealthy * highNegUtility

  return utility, (lowUtility, highUtility)

# %%
def analyzeTreeSample(tree, agents, confidence_level=0.95, bootstrap_samples=1000, cumulative_prob = 0.95, max_samples = 100000):

    agentDict = dict()

    for (id, utility, health) in agents:
        agentDict[id] = (utility, health)

    utilities = []
    probabilities = []

    probDict = dict()
    utilDict = dict()

    samples_taken = 0

    while sum(probDict.values()) < cumulative_prob and samples_taken < max_samples:

        samples_taken += 1

        negAgents = set()
        utility = 0
        probability = 1
        negTestedAgents = set()

        for agent in agents:

            id, _, health = agent
            # Determine if the agent is healthy based on their health probability
            is_healthy = random.random() < health

            # If the agent is unhealthy, add to the negative_agents list
            if is_healthy:
                negAgents.add(id)
                probability *= health
            else:
                probability *= 1 - health

        if frozenset(negAgents) in probDict:
            continue

        def exploreTree(tree):
            test, posScenario, negScenario = tree
            if test.issubset(negAgents):
                negTestedAgents.update(test)
                if negScenario:
                    exploreTree(negScenario)
            elif posScenario:
                exploreTree(posScenario)

        exploreTree(tree)

        probDict[frozenset(negAgents)] = probability

        for id in negTestedAgents:
            utility += agentDict[id][0]

        utilDict[frozenset(negAgents)] = utility

        utilities.append(utility)
        probabilities.append(probability)

    weighted_mean = 0

    orderedProbabilities = []
    orderedUtilities = []

    for instance in probDict:
        weighted_mean += probDict[instance] * utilDict[instance]
        orderedProbabilities.append(probDict[instance])
        orderedUtilities.append(utilDict[instance])

    orderedProbabilities = np.array(orderedProbabilities)
    pathsExplored = orderedProbabilities.sum()
    weighted_mean /= pathsExplored
    orderedProbabilities /= pathsExplored
    
    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(bootstrap_samples):
        # Resample with replacement, weighted by normalized probabilities
        sampled_utilities = np.random.choice(orderedUtilities, size=len(utilities), p=orderedProbabilities, replace=True)
        bootstrap_means.append(np.mean(sampled_utilities))
    
    # Calculate confidence intervals
    lower_percentile = (100 - confidence_level * 100) / 2
    upper_percentile = 100 - lower_percentile
    ci_lower, ci_upper = np.percentile(bootstrap_means, [lower_percentile, upper_percentile])
    
    return weighted_mean, (ci_lower, ci_upper), pathsExplored


# %% [markdown]
# ### RL Model

# %% [markdown]
# #### Training Model

# %%
# Define the number of agents and the group test parameters
N = 50  # Number of agents
B = 2  # Number of tests (defines output boolean list size)
G = 5
d = 3
classOrder = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
input_size = len(classOrder) # Two relevant attributes per agent
output_size = N  # Boolean list size
num_epochs = 500
num_samples = 100000
lr = 0.01
beta = 0.1
beta_decay = 0.000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(processName)s - %(message)s",
    handlers=[
        logging.FileHandler(f"training_N{N}_Seq_G{G}.log"),
        logging.StreamHandler()
    ]
)

def singleSameUtility(agents, G):
    # Sort by `num` in descending order
    agents.sort(key=lambda x: x[2] * x[1], reverse=True)
    
    max_value = 0
    best_subset_ids = set()
    current_product = 1
    current_count = 0

    for id, _, num in agents:
        if current_count == G:  # Stop if the subset size reaches G
            break

        # Calculate the new product and value
        new_product = current_product * num
        new_value = new_product * (current_count + 1)
        
        if new_value > max_value:
            # Update the current optimal solution
            max_value = new_value
            current_product = new_product
            current_count += 1
            best_subset_ids.add(id)
        else:
            break  # Stop adding smaller values when they don't improve the result

    return best_subset_ids, max_value

# %%
def load_RL_model_if_exists(model, samples, epoch, beta, beta_decay, lr, save_path='N50_Seq_Bucket_Util1_7Layers/'):
    model_filename = f"{save_path}dynamic_RL_model_N{N}_Seq_G{G}_B{B}_s{samples}_e{epoch}_b{beta}_lr{lr}_bd{beta_decay}.pth"
    if os.path.exists(model_filename):
        model.load_state_dict(torch.load(model_filename))
        print(f"Loaded model from {model_filename}", flush = True)
        logging.info(f"Loaded model from {model_filename}")
        return True
    return False

class SeqRLModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SeqRLModel, self).__init__()
        
        # 7 hidden layers:
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64, 32)
        
        self.fc_out = nn.Linear(32, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        
        x = torch.sigmoid(self.fc_out(x))  # Output probabilities
        return x

# Function to sample actions (boolean values) based on the predicted probabilities
# def sample_actions(probabilities):
#     # Mask values <= 0.5
#     mask = probabilities > 0.5
#     filtered_probs = probabilities * mask  # Set values <= 0.5 to 0

#     # Sort indices by descending probability
#     sorted_indices = torch.argsort(filtered_probs, descending=True)
#     sorted_probs = filtered_probs[sorted_indices]

#     # Find the G-th largest value
#     gth_value = sorted_probs[G - 1]

#     # Identify the indices with the G-th largest value (ties)
#     tie_indices = sorted_indices[sorted_probs == gth_value]

#     # Ensure the total selected is exactly G
#     if len(tie_indices) > 1:  # Ties exist at the G-th value
#         # Randomly shuffle tied indices and pick enough to reach exactly G
#         shuffle = torch.randperm(len(tie_indices))
#         tie_indices = tie_indices[shuffle]

#     # Combine all indices above the G-th value and randomly chosen ties
#     above_g_indices = sorted_indices[sorted_probs > gth_value]
#     final_indices = torch.cat((above_g_indices, tie_indices[: G - len(above_g_indices)]))

#     # Create a boolean output vector
#     selected = torch.zeros_like(probabilities, dtype=torch.bool)
#     selected[final_indices] = True

#     return selected

def sample_actions(probabilities, G=G):
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.detach().cpu()  # Ensure it's on CPU for sampling
    else:
        probabilities = torch.tensor(probabilities, dtype=torch.float32)  # Convert to tensor

    # Normalize probabilities and sample G indices without replacement
    selected_indices = torch.multinomial(probabilities, G, replacement=False)

    # Create a boolean mask of selected actions
    selected = torch.zeros_like(probabilities, dtype=torch.bool)
    selected[selected_indices] = True

    return selected

def create_RL_training_data(num_samples):
    # Example: create a list of agents
    return [create_agents(N) for _ in range(num_samples)]

# Training loop using REINFORCE with model saving
# Preprocess the agents by extracting the relevant attributes
def preprocess_agents(agents, classOrder=classOrder):
    # Extract the class tuples from agents
    # agent_classes = [tuple(agent[1:]) for agent in agents]
    
    # # Count occurrences of each class in classOrder
    # counts = [agent_classes.count(class_type) for class_type in classOrder]

    counts = [sum(a < agent[2] * agent[1] <= b for agent in agents) for a, b in classOrder]
    
    # Convert the counts list to a PyTorch tensor
    return torch.tensor(counts, dtype=torch.float32).flatten()

def nextAction(agents, newB):
  if newB == 1:
    ids, _ = singleSameUtility(agents, G)
    # action = {}
    # Create the boolean list
    return frozenset(ids) if ids is not None else frozenset()
  else:
    print("Error: B must be 1")
    return frozenset()

# Training loop using REINFORCE with model saving
def train_RL_model(model, optimizer, training_data, num_epochs, lr, beta, beta_decay, save_interval=100, save_path='N50_Seq_Bucket_Util1_7Layers/'):

    # Function to save the model
    def save_RL_model(model, samples, epoch, save_path):
        model_filename = f"{save_path}dynamic_RL_model_N{N}_Seq_G{G}_B{B}_s{samples}_e{epoch}_b{beta}_lr{lr}_bd{beta_decay}.pth"
        torch.save(model.state_dict(), model_filename)
        print(f"Model saved at epoch {epoch} to {model_filename}", flush = True)
        logging.info(f"Model saved at epoch {epoch} to {model_filename}")

    start_epoch = 0

    # Check if a model has already been trained for a lesser number of epochs
    for epoch in range(num_epochs, 0, -1):
        if load_RL_model_if_exists(model, len(training_data), epoch, beta, beta_decay, lr, save_path=save_path):
            start_epoch = epoch
            break

    # Now continue training from start_epoch to num_epochs
    for epoch in tqdm(range(start_epoch, num_epochs)):
        epoch_loss = 0
        epoch_reward = 0
        current_beta = beta * ((1-beta_decay) ** epoch)

        epoch_time = time.time()
        print(f"begin epoch {epoch}", flush=True)

        # print(f"Current beta: {current_beta}")

        for agents in training_data:
            # aTime = time.time()
            # print(f"per agent time: {aTime - start}", flush=True)
            # start = time.time()
            # Preprocess agents to get only the relevant attributes
            # now = time.time()
            # print("preprocess", flush=True)
            input_data = preprocess_agents(agents)
            input_data = input_data.unsqueeze(0)  # Add batch dimension
            # print(f"Input data shape: {input_data.shape}", flush=True)

            # B_tensor = torch.tensor([B], dtype=torch.float32, device=input_data.device).unsqueeze(0)  # Shape: (1, 1)
            # input_data = torch.cat((B_tensor, input_data), dim=-1) 
            # print(f"Input data shape: {input_data.shape}", flush=True)
            # print(f"{time.time()-now}, input", flush=True)
            # now = time.time()
            # Forward pass: predict probabilities for the boolean list
            predicted_probabilities = model(input_data)
            predicted_probabilities = torch.clamp(predicted_probabilities, 0.0, 1.0)

            # Sample actions based on the predicted probabilities
            # print(f"{time.time()-now}, sample", flush=True)
            # now = time.time()
            m = sample_actions(predicted_probabilities[0])

            # Convert sampled boolean list to a tree structure and compute utility (reward)

            choices = frozenset(np.where(m)[0])

            sorted_agents = sorted(agents, key=lambda x: x[2])

            # Step 2: Select IDs based on 0-based choices indices
            selection = {sorted_agents[x][0] for x in choices if 0 <= x < len(sorted_agents) and sorted_agents[x][2] > 0}
            selection = frozenset(selection)

            health_values = torch.tensor([agent[2] for agent in agents])

            # Perform independent Bernoulli trials
            bern_samples = torch.bernoulli(health_values).int()

            negative = all(bern_samples[[i for i, agent in enumerate(agents) if agent[0] in selection]] == 1)

            # print(f"{time.time()-now}, iters", flush=True)
            # now = time.time()

            actions = [selection]

            if negative:
              #### TODO for larger B, need to add a fix -- model should not want to include neg people in more tests, waste of space for G.
              # print(f"{time.time()-now}, b2", flush=True) 
              # now = time.time()
              negDict = bayesTheorem(agents, posGroups={}, negAgents=selection)
              negOutcomeAgents = sorted(
                  ((idNum, utility, 0 if utility == 0 else health) for idNum, (utility, health) in negDict.items()),
                  key=lambda x: x[0]
              )
              # process here, if tested and negative
              # print(f"{time.time()-now}, next", flush=True)
              # now = time.time()
              actions.append(nextAction(negOutcomeAgents, B-1))
            else:
              ########## todo for larger B, not use bayes, use gibbs mcmc for both. apply changes to eval
              # print(f"{time.time()-now}, bayes", flush=True)
              # now = time.time()
              posDict = bayesTheorem(agents, posGroups={selection}, negAgents={})
              posOutcomeAgents = sorted(((idNum, utility, health) for idNum, (utility, health) in posDict.items()), key=lambda x: x[0])
              # print(f"{time.time()-now}, next", flush=True)
              # now = time.time()
              actions.append(nextAction(posOutcomeAgents, B-1))

            # print(f"{time.time()-now}, tree", flush=True)
            # now = time.time()
            # print(f"{time.time()-now}, analyze", flush=True)
            # now = time.time()
            id_to_util = {agent[0]: (agent[1], i) for i, agent in enumerate(agents)}

            used_ids = set()  # To ensure each person's utility is counted once

            reward = 0
            for action in actions:
                # Find indices of selected IDs in bern_samples
                indices = [id_to_util[aid][1] for aid in action if aid in id_to_util]
                
                # Check if all are healthy (1 in bern_samples)
                if all(bern_samples[indices] == 1):
                    for aid in action:
                        if aid not in used_ids and aid in id_to_util:
                            reward += id_to_util[aid][0]  # Add utility
                            used_ids.add(aid)
            # print(f"Boolean tree: {boolean_tree}")
            # print(f"{time.time()-now}, opt", flush=True)
            # now = time.time()

            # Policy gradient loss (-log_prob * reward)
            m = torch.zeros(N, dtype=torch.float32)

            # Set indices in selection to 1
            indices = torch.tensor(list(selection), dtype=torch.long)
            m[indices] = 1.0
            log_probs_actions = m.float() * torch.log(predicted_probabilities + 1e-8) + (1 - m.float()) * torch.log(1 - predicted_probabilities + 1e-8)

            log_probs = log_probs_actions.sum(dim=1)  # Sum across the boolean list
            # print(f"Log probs: {log_probs}")

            # Compute entropy
            entropy = -torch.sum(predicted_probabilities * log_probs_actions, dim=1)

            # print(f"Entropy: {entropy}")
            # print(f"Reward {reward}")

            # Policy gradient loss (-log_prob * reward) 
            loss = -(log_probs * reward) + -(current_beta * entropy)

            # loss = loss.mean()

            # print(f"Loss: {loss}", flush = True)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            # for name, param in model.named_parameters():
            #   if param.grad is not None:  # Gradients might not exist for all params
            #       if torch.isnan(param.grad).any():
            #           print(f"NaN in gradient of {name} during training with agents: {agents}, reward: {reward}, loss: {loss}, log_probs: {log_probs}, current_beta: {current_beta}, entropy: {entropy}, probs: {predicted_probabilities}", flush=True)
            #       if torch.isinf(param.grad).any():
            #           print(f"Infinity in gradient of {name} during training with agents: {agents}, reward: {reward}, loss: {loss}, log_probs: {log_probs}, current_beta: {current_beta}, entropy: {entropy}, probs: {predicted_probabilities}", flush=True)

            optimizer.step()

            # for name, param in model.named_parameters():
            #   if torch.isnan(param).any():
            #       print(f"NaN in parameter: {name} after optimizer step with agents: {agents}, reward: {reward}, loss: {loss}, log_probs: {log_probs}, current_beta: {current_beta}, entropy: {entropy}, probs: {predicted_probabilities}", flush=True)
            #   if torch.isinf(param).any():
            #       print(f"Infinity in parameter: {name} after optimizer step with agents: {agents}, reward: {reward}, loss: {loss}, log_probs: {log_probs}, current_beta: {current_beta}, entropy: {entropy}, probs: {predicted_probabilities}", flush=True)

            epoch_loss += loss.item()
            epoch_reward += reward

        # Save model every `save_interval` epochs
        if (epoch + 1) % save_interval == 0:
            os.makedirs(save_path, exist_ok=True)
            model_filename = f"{save_path}dynamic_RL_model_N{N}_Seq_G{G}_B{B}_s{len(training_data)}_e{epoch + 1}_b{beta}_lr{lr}_bd{beta_decay}.pth"
            torch.save(model.state_dict(), model_filename)
            print(f"Saved model at {model_filename}", flush = True)
            logging.info(f"Saved model at {model_filename}")

        now = time.time()
        avg_loss = epoch_loss / len(training_data)
        avg_reward = epoch_reward / len(training_data)
        print(f"Epoch {epoch + 1}, LR: {lr}, Beta: {beta}, Beta Decay: {beta_decay}, Average Loss: {avg_loss:.4f}, Average Reward: {avg_reward:.4f}, {now - epoch_time} total seconds", flush = True)
        logging.info(f"Epoch {epoch + 1}, LR: {lr}, Beta: {beta}, Beta Decay: {beta_decay}, Average Loss: {avg_loss:.4f}, Average Reward: {avg_reward:.4f}, {now - epoch_time} total seconds")
    
    if num_epochs > start_epoch:
        save_RL_model(model, len(training_data), num_epochs, save_path)

# %%
def getAgentsFromTrainData(path):

    # Load the CSV into a DataFrame
    agentdf = pd.read_csv(path)

    # Convert the column of strings into Python objects (lists of tuples)
    agentdf['agents'] = agentdf['agents'].apply(ast.literal_eval)

    # Extract the column as a Python list of lists of tuples
    agents = agentdf['agents'].tolist()

    return agents

# %% [markdown]
# ##### Hyperparameter Tuning

# %% [markdown]
# Parallel Processing Hyperparameter Tuning

# %%
# Load training data
print("load", flush=True)
training_data = getAgentsFromTrainData(f'training_data_N{N}_Seq_B{B}_G{G}.csv')[:num_samples]
eval_data = getAgentsFromTrainData(f'eval_data_N{N}_Seq_B{B}_G{G}.csv')[:num_samples]
print(f"Loaded from training_data_N{N}_Seq_G{G}_B{B}_G{G}.csv for len {len(training_data)} and eval_data_N{N}_Seq_G{G}_B{B}_G{G}.csv for len {len(eval_data)}")
logging.info(f"Loaded from training_data_N{N}_Seq_G{G}_B{B}_G{G}.csv for len {len(training_data)} and eval_data_N{N}_Seq_G{G}_B{B}_G{G}.csv for len {len(eval_data)}")

# Define the scoring function (e.g., average reward over training data)
def evaluate_model(model, training_data):
    model.eval()  # Set model to evaluation mode
    total_reward = 0
    with torch.no_grad():
        for agents in training_data:
            input_data = preprocess_agents(agents)
            input_data = input_data.unsqueeze(0)  # Add batch dimension
            # print(f"Input data shape: {input_data.shape}", flush=True)

            # B_tensor = torch.tensor([B], dtype=torch.float32, device=input_data.device).unsqueeze(0)  # Shape: (1, 1)
            # input_data = torch.cat((B_tensor, input_data), dim=-1) 
            # print(f"Input data shape: {input_data.shape}", flush=True)
            # print(f"{time.time()-now}, input", flush=True)
            # now = time.time()
            # Forward pass: predict probabilities for the boolean list
            predicted_probabilities = model(input_data)
            predicted_probabilities = torch.clamp(predicted_probabilities, 0.0, 1.0)
            # print("here")
            # print(torch.bernoulli(predicted_probabilities).tolist())
            # boolean_tree = boolean_list_to_tree(
            #    torch.bernoulli(predicted_probabilities).tolist(), len(agents)
            # )
            m = sample_actions(predicted_probabilities[0])

            # Convert sampled boolean list to a tree structure and compute utility (reward)

            choices = frozenset(np.where(m)[0])

            sorted_agents = sorted(agents, key=lambda x: x[2])

            # Step 2: Select IDs based on 0-based choices indices
            selection = {sorted_agents[x][0] for x in choices if 0 <= x < len(sorted_agents) and sorted_agents[x][2] > 0}
            selection = frozenset(selection)

            health_values = torch.tensor([agent[2] for agent in agents])

            # Perform independent Bernoulli trials
            bern_samples = torch.bernoulli(health_values).int()

            negative = all(bern_samples[[i for i, agent in enumerate(agents) if agent[0] in selection]] == 1)

            # print(f"{time.time()-now}, iters", flush=True)
            # now = time.time()

            actions = [selection]

            if negative:
              #### TODO for larger B, need to add a fix -- model should not want to include neg people in more tests, waste of space for G.
              # print(f"{time.time()-now}, b2", flush=True) 
              # now = time.time()
              negDict = bayesTheorem(agents, posGroups={}, negAgents=selection)
              negOutcomeAgents = sorted(
                  ((idNum, utility, 0 if utility == 0 else health) for idNum, (utility, health) in negDict.items()),
                  key=lambda x: x[0]
              )
              # process here, if tested and negative
              # print(f"{time.time()-now}, next", flush=True)
              # now = time.time()
              actions.append(nextAction(negOutcomeAgents, B-1))
            else:
              ########## todo for larger B, not use bayes, use gibbs mcmc for both. apply changes to eval
              # print(f"{time.time()-now}, bayes", flush=True)
              # now = time.time()
              posDict = bayesTheorem(agents, posGroups={selection}, negAgents={})
              posOutcomeAgents = sorted(((idNum, utility, health) for idNum, (utility, health) in posDict.items()), key=lambda x: x[0])
              # print(f"{time.time()-now}, next", flush=True)
              # now = time.time()
              actions.append(nextAction(posOutcomeAgents, B-1))

            # print(f"{time.time()-now}, tree", flush=True)
            # now = time.time()
            # print(f"{time.time()-now}, analyze", flush=True)
            # now = time.time()
            id_to_util = {agent[0]: (agent[1], i) for i, agent in enumerate(agents)}

            used_ids = set()  # To ensure each person's utility is counted once

            reward = 0
            for action in actions:
                # Find indices of selected IDs in bern_samples
                indices = [id_to_util[aid][1] for aid in action if aid in id_to_util]
                
                # Check if all are healthy (1 in bern_samples)
                if all(bern_samples[indices] == 1):
                    for aid in action:
                        if aid not in used_ids and aid in id_to_util:
                            reward += id_to_util[aid][0]  # Add utility
                            used_ids.add(aid)
            total_reward += reward
    return total_reward / len(training_data)  # Return average reward

# Function to train and evaluate the model for a specific parameter combination
def train_and_evaluate(params, num_epochs, training_data, eval_data):
    print(params, flush=True)
    logging.info(f"Current params: {params}")
    lr, beta, beta_decay = params
    
    # Reinitialize the model and optimizer for each combination
    model = SeqRLModel(input_size=input_size, output_size=output_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Train the model
    train_RL_model(model, optimizer, training_data, num_epochs, lr, beta, beta_decay)
    
    # Evaluate the model
    avg_reward = evaluate_model(model, eval_data)
    
    # Return the result
    return {"lr": lr, "beta": beta, "num_epochs": num_epochs, "bea_decay": beta_decay, "avg_reward": avg_reward}

# Define the parameter grid
param_grid = {
    "lr": [0.001, 0.01, 0.05, 0.1],
    "beta": [0, 1.0],
    "beta_decay" : [0, 0.1]
}

num_epochs = [10, 25, 100, 500, 1000]
num_epochs.sort()

# Get all combinations of parameters
param_combinations = list(product(*param_grid.values()))
param_combinations = [p for p in param_combinations if not (p[1] == 0 and p[2] != 0)]

results = []
for epochs in num_epochs:
    batch_results = Parallel(n_jobs=-1, verbose=10)(
        delayed(train_and_evaluate)(params, epochs, training_data, eval_data) for params in param_combinations
    ) or []  # Ensure it's a list
    results.extend(batch_results)  # Avoid NoneType error

# Convert results to a DataFrame for better visualization
df_results = pd.DataFrame(results)
df_results.sort_values(by="avg_reward", ascending=False, inplace=True)

# Save results to CSV
df_results.to_csv(f'RLtuning_N{N}_B{B}_G{G}.csv', index=False)

# Find the best parameter combination
best_params = df_results.iloc[0]
print("Best Parameters:", best_params)
logging.info(f"Best params: {best_params}")
print(df_results)
logging.info(df_results)


