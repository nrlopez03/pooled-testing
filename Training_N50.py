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

# Example usage:
tree = ({0, 1}, ({1}, ({1}, None, None), ({1}, None, None)), ({2}, ({1}, None, None), ({1}, None, None)))

boolean_list = tree_to_boolean_list(tree, N)


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

# Example usage:
boolean_list = [True, True, False, False, True, False, False, True, False, False, True, False, False, False, True, False, True, False, False, True, False]
N = 3
reconstructed_tree = boolean_list_to_tree(boolean_list, N)


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
maxUtil(agents)

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
strategy, utility = solveStaticNonOverlap(agents)
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

# %% [markdown]
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
    

# %% [markdown]
# Custom Testing

# %%
agents = [(0, 1, 1/2), (1, 1, 1/2), (2, 1, 1/2)]
posGroups = [frozenset({0,1}), frozenset({1,2})]
negAgents = {}
# printBayes(bayesTheorem(agents, posGroups, negAgents))
# print("\n")
# printBayes(GibbsMCMCWindow(agents, posGroups, negAgents))
# print("\n")
evaluateCI(GibbsMCMCWindow(agents, posGroups, negAgents), bayesTheorem(agents, posGroups, negAgents))

# %% [markdown]
# Testing Overlap Results

# %% [markdown]

# %%
agents = [(i + 1, 1/2, 1/2) for i in range(9)]
posGroups = [frozenset({7, 3, 1, 4}), frozenset({2, 1, 5, 6})]
negAgents = {}
# printBayes(bayesTheorem(agents, posGroups, negAgents))
# print("\n")
# printBayes(GibbsMCMCWindow(agents, posGroups, negAgents))
GibbsMCMCWindow(agents, posGroups, negAgents)
# print("\n")
evaluateCI(GibbsMCMCWindow(agents, posGroups, negAgents), bayesTheorem(agents, posGroups, negAgents))

# %%
agents = [(i + 1, 1/2, 1/2) for i in range(9)]
posGroups = [frozenset({7, 3, 1, 4}), frozenset({2, 1, 5, 6}), frozenset({1})]
negAgents = {}
# printBayes(bayesTheorem(agents, posGroups, negAgents))
# print("\n")
# printBayes(GibbsMCMCWindow(agents, posGroups, negAgents))
# print("\n")
evaluateCI(GibbsMCMCWindow(agents, posGroups, negAgents), bayesTheorem(agents, posGroups, negAgents))

# %%
agents = [(i + 1, 1/2, 1/2) for i in range(9)]
posGroups = [frozenset({7, 3, 1, 4}), frozenset({2, 1, 5, 6}), frozenset({1}), frozenset({8, 6}), frozenset({9, 7})]
negAgents = {}
# printBayes(bayesTheorem(agents, posGroups, negAgents))
# print("\n")
# printBayes(GibbsMCMC(agents, posGroups, negAgents))
# print("\n")
evaluateCI(GibbsMCMCWindow(agents, posGroups, negAgents), bayesTheorem(agents, posGroups, negAgents))

# %%
agents = [(i + 1, 1/2, 1/2) for i in range(9)]
posGroups = [frozenset({1}), frozenset({8, 6}), frozenset({9, 7})]
negAgents = {}
# printBayes(bayesTheorem(agents, posGroups, negAgents))
# print("\n")
# printBayes(GibbsMCMCWindow(agents, posGroups, negAgents))
# print("\n")
evaluateCI(GibbsMCMCWindow(agents, posGroups, negAgents), bayesTheorem(agents, posGroups, negAgents))

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

# %%
agents = [(0, 1, 0.5), (1, 1, 0.5), (2, 1, 1)]
staticStrategy, staticUtility = solveStaticOverlap(agents)
# printStatic(staticStrategy, staticUtility)

# %%
# print(staticStrategy)

# %%
agents = [(0, 0.022080006061812485, 0.8205655186482051), (1, 0.760970132052074, 0.5930101907614745), (2, 0.42354957403482274, 0.6984521403961065), (3, 0.006600413727435472, 0.2859953353854493), (4, 0.616360236705176, 0.29752798376295153)]
staticStrategy, staticUtility = solveStaticOverlap(agents, B= 4)
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

# %%
agents = [(0, 1, 0.5), (1, 1, 0.5), (2, 1, 1)]
dynamicStrategy, dynamicUtility = solveDynamic(agents)

# printDynamic(dynamicStrategy, dynamicUtility)

# %% [markdown]
# ### Dynamic Strategy Analysis and Representation

# %%
agents = [(0, 0.129, 0.5562), (1, 0.17483, 1), (2, 0.569, 0.12)]

dynamicStrategy, dynamicUtility = solveDynamic(agents)

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
B = 10  # Number of tests (defines output boolean list size)
G = 5
d = 3
num_types = 9
input_size = num_types  # Two relevant attributes per agent
num_tests = 2 ** (B -1)
output_size = num_tests * G  # Boolean list size
num_epochs = 1000
num_samples = 100000
lr = 0.01
beta = 0.1
beta_decay = 0.000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(processName)s - %(message)s",
    handlers=[
        logging.FileHandler(f"training_N{N}.log"),
        logging.StreamHandler()
    ]
)


# %%
def load_RL_model_if_exists(model, samples, epoch, beta, beta_decay, lr, save_path='models/'):
    model_filename = f"{save_path}dynamic_RL_model_N{N}_B{B}_s{samples}_e{epoch}_b{beta}_lr{lr}_bd{beta_decay}.pth"
    if os.path.exists(model_filename):
        model.load_state_dict(torch.load(model_filename))
        print(f"Loaded model from {model_filename}", flush = True)
        logging.info(f"Loaded model from {model_filename}")
        return True
    return False

class GroupedAgentIDModel(nn.Module):
    def __init__(self, input_size, num_tests):
        super(GroupedAgentIDModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_tests * G)  # Predict G IDs for each group

        self.num_tests = num_tests
        self.group_size = G
        self.N = N

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        # Reshape to (batch_size, num_groups, group_size)
        return logits.view(-1, self.num_tests, G)


# Function to sample actions (boolean values) based on the predicted probabilities
def get_agent_ids(logits):
    """
    Convert logits into discrete agent IDs.
    
    Args:
        logits (torch.Tensor): Logits of shape (batch_size, num_groups, group_size).
        num_agents (int): Total number of agents (N).
    
    Returns:
        torch.Tensor: Predicted agent IDs of shape (batch_size, num_groups, group_size).
    """
    # Ensure logits are integers in [0, num_agents - 1]
    agent_ids = torch.round(logits).long() % N
    return agent_ids

def create_RL_training_data(num_samples):
    # Example: create a list of agents
    return [create_agents(N) for _ in range(num_samples)]

# Training loop using REINFORCE with model saving
# Preprocess the agents by extracting the relevant attributes
# todo
def preprocess_agents(agents):
    processed_agents = [agent[1:] for agent in agents]  # Ignore the first attribute
    return torch.tensor(processed_agents, dtype=torch.float32).flatten()  # Flatten for input

# Training loop using REINFORCE with model saving
def train_RL_model(model, optimizer, training_data, num_epochs, lr, beta, beta_decay, save_interval=100, save_path='models/'):

    # Function to save the model
    def save_RL_model(model, samples, epoch, save_path):
        model_filename = f"{save_path}dynamic_RL_model_N{N}_B{B}_s{samples}_e{epoch}_b{beta}_lr{lr}_bd{beta_decay}.pth"
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
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_reward = 0

        # Decay beta
        current_beta = beta * ((1 - beta_decay) ** epoch)

        for agents in training_data:
            # Preprocess agents
            input_data = preprocess_agents(agents)
            input_data = input_data.unsqueeze(0)  # Add batch dimension

            # Forward pass
            logits = model(input_data)

            # Get discrete agent IDs
            predicted_ids = get_agent_ids(logits)

            # Compute reward based on predicted IDs
            reward = compute_reward(predicted_ids, agents)

            # Compute probabilities for entropy
            probabilities = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8))  # Avoid log(0)

            # Combine reward and entropy
            loss = -(reward.mean() + current_beta * entropy)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_reward += reward.item()

        # Save model every `save_interval` epochs
        if (epoch + 1) % save_interval == 0:
            os.makedirs(save_path, exist_ok=True)
            model_filename = f"{save_path}dynamic_RL_model_N{N}_B{B}_s{len(training_data)}_e{epoch + 1}_b{beta}_lr{lr}_bd{beta_decay}.pth"
            torch.save(model.state_dict(), model_filename)
            print(f"Saved model at {model_filename}", flush = True)
            logging.info(f"Saved model at {model_filename}")

        avg_loss = epoch_loss / len(training_data)
        avg_reward = epoch_reward / len(training_data)
        print(f"Epoch {epoch + 1}, LR: {lr}, Beta: {beta}, Beta Decay: {beta_decay}, Average Loss: {avg_loss:.4f}, Average Reward: {avg_reward:.4f}", flush = True)
        logging.info(f"Epoch {epoch + 1}, LR: {lr}, Beta: {beta}, Beta Decay: {beta_decay}, Average Loss: {avg_loss:.4f}, Average Reward: {avg_reward:.4f}")
    
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
training_data = getAgentsFromTrainData(f'training_data_N{N}_B{B}_G{G}.csv')
eval_data = getAgentsFromTrainData(f'eval_data_N{N}_B{B}_G{G}.csv')
print(f"Loaded from training_data_N{N}_B{B}_G{G}.csv and eval_data_N{N}_B{B}_G{G}.csv")
logging.info(f"Loaded from training_data_N{N}_B{B}_G{G}.csv and eval_data_N{N}_B{B}_G{G}.csv")

# Define the scoring function (e.g., average reward over training data)
def evaluate_model(model, training_data):
    model.eval()  # Set model to evaluation mode
    total_reward = 0
    with torch.no_grad():
        for agents in training_data:
            input_data = preprocess_agents(agents).unsqueeze(0)
            logits = model(input_data)

            predicted_ids = get_agent_ids(logits)

            # Compute reward based on predicted IDs
            reward = compute_reward(predicted_ids, agents)

            total_reward += reward
    return total_reward / len(training_data)  # Return average reward

# Function to train and evaluate the model for a specific parameter combination
def train_and_evaluate(params, training_data, eval_data):
    print(params)
    logging.info(f"Current params: {params}")
    lr, beta, num_epochs, beta_decay = params
    
    # Reinitialize the model and optimizer for each combination
    model = GroupedAgentIDModel(input_size, num_tests)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Train the model
    train_RL_model(model, optimizer, training_data, num_epochs, lr, beta, beta_decay)
    
    # Evaluate the model
    avg_reward = evaluate_model(model, eval_data)
    
    # Return the result
    return {"lr": lr, "beta": beta, "num_epochs": num_epochs, "bea_decay": beta_decay, "avg_reward": avg_reward}

# Define the parameter grid
param_grid = {
    "lr": [0.001, 0.01, 0.05, 0.1],
    "beta": [0.01, 0.1, 0.5, 1.0],
    "num_epochs": [10, 100, 500, 1000],
    "beta_decay" : [0, 0.0001, 0.01, 0.1]
}

# Get all combinations of parameters
param_combinations = list(product(*param_grid.values()))
param_combinations = sorted(param_combinations, key=lambda x: x[2])

# Use joblib to parallelize the grid search
results = Parallel(n_jobs=-1, verbose=10)(
    delayed(train_and_evaluate)(params, training_data, eval_data) for params in param_combinations
)

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

