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


B = 2
k = 2
N = 3
G = N


def create_agents(N = N, uIntegers=False):
  agents = list()

  for i in range(N):
    if uIntegers:
      agents.append((i, round(random.random() * 100) + 1, random.random()))
    else:
      agents.append((i, random.random(), random.random()))
  return agents

agents = create_agents()


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


def boolean_list_to_tree(bool_list, N):

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


def maxUtil(agents):
  util = 0
  for agent in agents:
    util += agent[1] * agent[2]
  return util

def solveStaticNoPool(agents, B=B):
  utils = []
  for agent in agents:
    utils.append(agent[1] * agent[2])
  utils = sorted(utils, reverse=True)
    # Take the first B elements and sum them
  util = sum(utils[:B])
  return util


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
    

def solveMILP(agents, G=G, B=B):

    def approx_model(q, u, n, T=B, G=G, K=20):

        """
        Build cluster-based MILP model for Gurobi to solve. Finds an approximately
        optimal testing allocation.
        Inputs are three vectors indexed by cluster, as well as number of tests T
        and pool size bound G, and accuracy parameter K.

        q::Vector - avg. probability of being healthy for each cluster
        u::Vector - utility for each cluster
        n::Vector - size of each cluster
        T::Int 	  - number of tests
        G::Int    - pooled test size
        K::Int 	  - number of segments of piecewise-linear fn approximating exp constraint
        """
        
        # Verify that input is consistent
        assert len(u) == len(q) == len(n), "Input vectors have different lengths."
        assert K >= 1, "Number of segments for approximating exp must be at least 1."
        assert T <= sum(n), "Number of tests cannot exceed number of people in population."
        assert all(isinstance(x, int) and x > 0 for x in u), "Utilities must be strictly positive integers."
        assert all(0 <= x <= 1 for x in q), "Probabilities must (strictly) lie between 0 and 1."

        # Compute some constants
        C = len(n)  # number of clusters C
        # Lower and upper bounds for z[t] = x[t]⋅u
        L, U = min(u), G*max(u)
        # print(f"L: {L}, U: {U}")
        # Lower and upper bounds for l[t] = log(x[t]⋅u) + sum(x[t,i]*log(q[i])
        A = min(log(x) for x in u) + G*min(log(x) for x in q)
        B = log(G*max(u)) + max(log(x) for x in q)
        # print(f"A: {A}, B: {B}")
        tests = range(0, T)
        clusters = range(0, C)
        segments = range(0,K)

        # Create model and set parameters
        m = gp.Model('Test Allocation')
        m.setParam('OutputFlag', 0)
        # m.setParam("TimeLimit", 600)
        # m.setParam("Presolve", -1)
        m.setParam('MIPGap', 0.01)

        # Define variables
        x = m.addVars(tests, clusters, lb = 0, vtype = GRB.INTEGER, name='x')
        w = m.addVars(tests, lb=0, name='w')
        l = m.addVars(tests, lb=-GRB.INFINITY, name='l')
        y = m.addVars(tests, lb=-GRB.INFINITY, name='y')
        z = m.addVars(tests, lb=-GRB.INFINITY, name='z')
        # variables for log constraint
        zind = m.addVars(tests, range(L,U+1), vtype=GRB.BINARY, name='zind')
        # variables for approximating exp constraint
        lind = m.addVars(tests, segments, vtype=GRB.BINARY, name='lind')
        v = m.addVars(tests, segments, lb=-GRB.INFINITY, name='v')

        # Set objective
        m.setObjective(sum(w[t] for t in tests), GRB.MAXIMIZE)

        # Add constraints
        m.addConstrs(sum(x[t,i] for t in tests) <= n[i] for i in clusters)  # tests must be disjoint
        m.addConstrs(1 <= sum(x[t,i] for i in clusters) for t in tests)  # pool size >= 1
        m.addConstrs(sum(x[t,i] for i in clusters) <= G for t in tests)  # pool size <= G

        # Log welfare constraints: l[t] == log(u ̇x[t]) + x[t] ̇log.(q)
        m.addConstrs(l[t] == y[t] + sum(x[t,i] * log(q[i]) for i in clusters) for t in tests)

        # Constraints to ensure y[t] <= log(z[t])
        m.addConstrs(z[t] == sum(x[t,i] * u[i] for i in clusters) for t in tests)  # utility sums
        # Use indictator variables t capture value of z[t]:
        # z[t] is an integer in [L, U], so let zind[t,k] = 1 if z[t] = k and 0 otherwise.
        m.addConstrs(1==sum(zind[t,k] for k in range(L,U+1)) for t in tests)  # exactly one zind entry is 1
        m.addConstrs(z[t] == sum(k*zind[t,k] for k in range(L,U+1)) for t in tests)
        m.addConstrs(y[t] <= sum(log(k)*zind[t,k] for k in range(L,U+1)) for t in tests)

        # Deal with w[t] = exp(l[t])
        if abs(B-A) < 1e-10:
            m.addConstrs(l[t] == A for t in tests)
            m.addConstrs(w[t] == exp(A) for t in tests)
        else:
            # Approximate w[t] <= exp(l[t]) using piecewise-linear function f with K segments on domain [A,B]
            c = optimal_partition(A, B, K)  # compute optimal segmentation of interval [A, B]
            a, b, c = linearise(exp, c)  # compute piecewise-linear function f on domain [A, B] with segmentation c
            # print(a, b, c)
            # Use indicator variables `lind[t,k]` to capture in which segment the value of l[t] lies
            # and let v[t,k] = l[t] if l[t] lies in (c[k], c[k+1]) and v[t,k] = 0 otherwise.
            m.addConstrs(1 == sum(lind[t,k] for k in segments) for t in tests)
            m.addConstrs(c[k]*lind[t,k] <= v[t,k] for t in tests for k in segments)
            m.addConstrs(v[t,k] <= c[k+1]*lind[t,k] for t in tests for k in segments)
            m.addConstrs(l[t] == sum(v[t,k] for k in segments) for t in tests)
            # Ensure that w[t] <= f(l[t])
            m.addConstrs(w[t] <= sum(a[k]*v[t,k] + b[k]*lind[t,k] for k in segments) for t in tests)

        # Return model
        return m, x

    def linearise(f, c):
        """
        Compute the piecewise-linear representation of `f` with segments specified
        in vector `c`.
        """
        K = len(c)-1  # number of segments
        a, b = np.zeros(K), np.zeros(K)
        for k in range(0,K):
            a[k] = (f(c[k+1]) - f(c[k])) / (c[k+1] - c[k])  # determine slope
            b[k] = f(c[k+1]) - a[k]*c[k+1]  # determine residual
        return a, b, c

    def delta(l, r):
        """
        Compute maximum difference between segment (l, exp(l)) to (r, exp(r))
        and exp(x) on the interval [l,r].
        """
        if r <= l: return 0.0
        a = (exp(l) - exp(r)) / (l - r)
        if a == 0: return 0.0  # happens if l and r are sufficiently similar
        b = exp(r) - a*r
        result = a * log(a) + b - a  # maximum difference, derived from first order conditions
        return max(0, result)  # slight hack to avoid numerical inaccuracies

    def partition(A, K, r1):
        """
        Build a partition of K segments starting from A such that the
        first segment is [A, A+r1] and all segments have the same error
        ε identical to the error of the first segment.
        """
        assert r1 >= 0
        assert K > 0
        c = [A]*(K+1)
        if r1 == 0: return c
        eps = delta(A, A + r1)  # error of the first segment [Lo, r1]
        for k in range(0,K):
            l = c[k]
            # To define the bracket for the root finder, we make the reasonable
            # assumption that the interval will be no larger than r1. (This can
            # be proved easily, I believe).
            r = brentq(lambda x : delta(l,x)-eps, l, l+r1+1)  # Finds r such that Δ(l,r) = ε.
            c[k+1] = r
        return c

    def optimal_partition(A, B, K):
        """
        Find the optimal partition of [A, B] into K segments. Proceeds by searching
        for the right size for the first segment: the size `r1` is right when
        `partition(A, K, r1)` ends (approximately) at `B`.
        """
        assert A < B
        first = brentq(lambda x : partition(A, K, x)[-1]-B, 0, B-A+1)
        c = partition(A, K, first)
        c[K] = B  # to clean things up a bit
        return c

    def compute_error(a,b,c):
        """ Compute the maximum difference between the segments of the piecewise-linear function f(x) specified by a, b,
        c and exp(x).

        NB: For segment k, the difference is maximised at x = log(a[k]).
        """
        ε = np.zeros(len(a))
        for k in range(2,len(a)):
            ε[k] = a[k]*np.log(a[k]) + b[k] - a[k]
        return max(ε)

    # u = [agent[1] for agent in agents]
    # q = [agent[2] for agent in agents]
    # n = [1] * len(agents)
    # m, x = approx_model(q, u, n, T=B, G=G)
    # m.optimize()

    # Group agents by (utility, health) and count them
    grouped_agents = defaultdict(lambda: 0)  # Dictionary to store count

    for agent in agents:
        key = (agent[1], agent[2])  # (utility, health)
        grouped_agents[key] += 1  # Increment the count of agents in this group

    u = []
    q = []
    n = []

    # Extract the grouped data
    for (utility, health), count in grouped_agents.items():
        u.append(utility)
        q.append(health)
        n.append(count)

    # Initialize an empty list to store the strategy

    strategy = []

    m, x = approx_model(q, u, n, T=B, G=G)
    m.optimize()

    # Loop over each test
    for test in range(B):
        # Collect the agents assigned to this test
        agents_assigned = []

        for cluster in range(len(u)):
            if x[test, cluster].X > 0:  # Only include agents assigned to this test
                # Append (u[cluster], q[cluster], n[cluster]) x[test, cluster].X times
                agents_assigned.extend([(cluster, u[cluster], q[cluster])] * int(x[test, cluster].X))

        strategy.append(tuple(agents_assigned))

    # Convert the strategy list to a tuple of tuples
    strategy = tuple(strategy)

    utility = m.objVal

    return strategy, utility

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
def analyzeTreeSample(tree, agents, confidence_level=0.95, bootstrap_samples=1000, cumulative_prob = 0.95, max_samples = 100000, unWeight=False, healthStatus = None):

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
            is_healthy = healthStatus[id] if healthStatus else random.random() < health

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

    if unWeight:
        return weighted_mean, (ci_lower, ci_upper), pathsExplored, next(iter(utilDict.values()))
    else:
        return weighted_mean, (ci_lower, ci_upper), pathsExplored


# %% [markdown]
# ### RL Model

# %% [markdown]
# #### Training Model
### adapted from https://github.com/edwinlock/csef/tree/main/optimisation/python

def solveConicGibbsGreedyDynamic(agents, G = G, B = B, posGroups = frozenset(), negAgents = frozenset(), CI = False, confidence_level=0.95):

  if B == 0:
    if CI:
      return [], 0, (0, 0)
    else:
        return [], 0

  agentDict = GibbsMCMCWindow(agents, posGroups, negAgents, confidence_level=confidence_level) 

  updatedAgents = [(id, utility, health, bounds) for id, (utility, health, bounds) in agentDict.items()]

  firstTest, _ = solveConicSingle(updatedAgents, G=G)

  agentDict = GibbsMCMCWindow(agents, posGroups, negAgents, confidence_level=confidence_level) 

  utility = 0

  if CI: 
    lowUtility = 0
    highUtility = 0

  firstUtility = 0
  firstHealthy = 1

  if CI: 
    lowFirstHealthy = 1
    highFirstHealthy = 1

  firstIDs = frozenset({person[0] for person in firstTest})

  remaining = [person for person in agents if person[0] not in firstIDs]

  if posGroups:
    for posGroup in posGroups:
      if posGroup.issubset(firstIDs):
        firstHealthy = 0
  else:
    posGroups = frozenset()

  for person in firstIDs:
    firstUtility += agentDict[person][0]
    firstHealthy *= agentDict[person][1]

    if CI: 
      lowFirstHealthy *= agentDict[person][2][0]
      highFirstHealthy *= agentDict[person][2][1]

  utility += firstUtility * firstHealthy

  if CI: 
    lowUtility += firstUtility * lowFirstHealthy
    highUtility += firstUtility * highFirstHealthy

  # positive scenario
  if firstHealthy < 1:

    newPosGroups = set(posGroups.copy())
    newPosGroups.add(firstIDs.difference(negAgents))

    if CI:

      posStrategy, posUtility, (lowPosUtility, highPosUtility) = solveConicGibbsGreedyDynamic(agents, G, B-1, frozenset(newPosGroups), frozenset(negAgents), CI, confidence_level=confidence_level)

    else:

      posStrategy, posUtility = solveConicGibbsGreedyDynamic(agents, G, B-1, frozenset(newPosGroups), frozenset(negAgents), CI)

    utility += (1 - firstHealthy) * posUtility

    if CI:

      lowUtility += (1 - firstHealthy) * lowPosUtility
      highUtility += (1 - firstHealthy) * highPosUtility

  else:

    posStrategy = []

  # negative scenario

  if remaining and firstHealthy > 0:
    newPosGroups = set()
    for posGroup in posGroups:
      newPosGroups.add(posGroup.difference(firstIDs))

    newNegAgents = set(negAgents.copy())
    newNegAgents.update(firstIDs)

    if CI:

      negStrategy, negUtility, (lowNegUtility, highNegUtility) = solveConicGibbsGreedyDynamic(remaining, G, B-1, frozenset(newPosGroups), frozenset(newNegAgents), CI, confidence_level=confidence_level)

    else:

      negStrategy, negUtility = solveConicGibbsGreedyDynamic(remaining, G, B-1, frozenset(newPosGroups), frozenset(newNegAgents), CI)

    utility += firstHealthy * negUtility

    if CI:

      lowUtility += firstHealthy * lowNegUtility
      highUtility += firstHealthy * highNegUtility

  else:

    negStrategy = []

  strategy = (tuple(firstTest), tuple(posStrategy), tuple(negStrategy))

  if CI:
    return strategy, utility, (lowUtility, highUtility)
  else:
    return strategy, utility
  
def strategy_to_tree(elements):
    if not elements:
        return "_"
    
    root = elements[0]
    left_subtree = strategy_to_tree(elements[1:]) if len(elements) > 1 else set()
    right_subtree = strategy_to_tree(elements[1:]) if len(elements) > 1 else set()

    return (root, left_subtree, right_subtree)

def eval_nonoverlap_sample(strategy, agents, healthStatus):
    """
    Computes the weighted sum of health status and u values for each pool of indices.
    
    :param strategy: List of lists, where each inner list is a pool of indices.
    :param agents: List of tuples (index, u, _).
    :param healthStatus: List of binary values (0 or 1) indicating health status.
    :return: The computed sum.
    """
    # Convert agents into a dictionary for quick lookup
    u_dict = {index: u for index, u, _ in agents}
    
    # Compute the total sum
    total_sum = sum(
        sum(u_dict[i] for (i, _, _) in pool) * prod(healthStatus[i] for (i, _, _) in pool)
        for pool in strategy
    )
    
    return round(total_sum)

Gvals = [3]
Bvals = [2,3,4]

for G in Gvals:
  for B in Bvals:
    file_path = f"sample_N50_d2_B{B}_G{G}_Utils3.csv"
    df = pd.read_csv(file_path)

    function = solveMILP

    tqdm.pandas()

    # Add an empty column if not present for storing results
    if function.__name__ not in df.columns:
        df[function.__name__] = None

    from joblib import Parallel, delayed
    from tqdm import tqdm
    import ast
    import pandas as pd
    import numpy as np

    # Define the column name to check for missing values
    output_column = function.__name__

    def process_row(i, agent_data, healthStatus):
        strategy, _ = function(agent_data, G=G, B=B) # type: ignore
        utilityResult = eval_nonoverlap_sample(strategy, agent_data, healthStatus)
        return i, utilityResult # type: ignore

    batch_size = 10
    n_jobs = -1  # Use all CPU cores

    # Identify rows that need processing
    rows_to_process = df[df[output_column].isna()].index.tolist()

    for batch_start in tqdm(range(0, len(rows_to_process), batch_size), desc="Processing batches", unit="batch"):
        batch_indices = rows_to_process[batch_start: batch_start + batch_size]
        
        row_data = [(i, ast.literal_eval(df.at[i, 'agents']), ast.literal_eval(df.at[i, 'healthStatus'])) for i in batch_indices]

        results = Parallel(n_jobs=n_jobs, backend="multiprocessing", verbose=10)(
            delayed(process_row)(i, data, healthStatus) for i, data, healthStatus in row_data
        )
        
        # Update DataFrame
        for i, value in results:
            df.at[i, output_column] = value

        df.to_csv(file_path, index=False)
        print(f"Saved at row {batch_start + batch_size}", flush=True)
