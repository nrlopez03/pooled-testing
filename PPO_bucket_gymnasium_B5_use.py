import gymnasium as gym
import numpy as np
import os
from gymnasium import spaces
from stable_baselines3 import PPO
import math
from mosek.fusion import * # type: ignore
from itertools import combinations, product, chain
import random 
import pandas as pd
import ast

N = 50
B = 5
G = 3
health_buckets = 4
more = True
eps = 50000000 if more else 20000000
direct_path_B2 = f"PPO_{health_buckets}Bucket_G{G}_More/PPO_model_N{N}_B2_G{G}_{eps}.zip" if more else f"PPO_{health_buckets}Bucket_G{G}/PPO_model_N{N}_B2_G{G}_{eps}.zip"
direct_path_B3 = f"PPO_{health_buckets}Bucket_G{G}_More/PPO_model_N{N}_B3_G{G}_{eps}.zip" if more else f"PPO_{health_buckets}Bucket_G{G}/PPO_model_N{N}_B3_G{G}_{eps}.zip"
direct_path_B4 = f"PPO_{health_buckets}Bucket_G{G}_More/PPO_model_N{N}_B4_G{G}_{eps}.zip" if more else f"PPO_{health_buckets}Bucket_G{G}/PPO_model_N{N}_B4_G{G}_{eps}.zip"
direct_path_B5 = f"PPO_{health_buckets}Bucket_G{G}_More/PPO_model_N{N}_B5_G{G}_{eps}.zip" if more else f"PPO_{health_buckets}Bucket_G{G}/PPO_model_N{N}_B5_G{G}_{eps}.zip"
file_path = f"sample_N50_d2_B{B}_G{G}_Utils3.csv"
column_name = f"PPO_{health_buckets}Bucket_{eps}"

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
    agentDict[agent[0]] = (agent[1], max(min(agent[2], 1), 0)) # reduce floating point error

  return agentDict

def solveConicSingle(agents, G, verbose=False):

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

        M.constraint("pool", Expr.sum(x), Domain.lessThan(G))

        M.solve()

        utility = math.exp(M.primalObjValue())
        strategy = tuple(agent for agent, level in zip(agents, np.round_(x.level())) if level == 1)

        return strategy, utility

class AgentSelectionEnvB2(gym.Env):
    def __init__(self, health_bins=health_buckets, utility_bins=3, max_selection=G, num_agents=N):
        super(AgentSelectionEnvB2, self).__init__()

        self.health_bins = health_bins
        self.utility_bins = utility_bins
        self.max_selection = max_selection
        self.num_agents = num_agents

        self.num_categories = health_bins * utility_bins
        self.observation_space = spaces.Box(low=0, high=100, shape=(self.num_categories + 2,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_categories + 1)  # categories + 1 stop action

        self.health_bin_edges = np.linspace(0, 1, health_bins + 1)
        self.utility_bin_edges = np.array([0, 2, 3])

        self.reset()

    def reset(self, agent_list=None, posGroups=None, negAgents=None, seed=None):
        """Initialize a new episode with a given set of 50 agents or generate a new set."""
        self.agents = []
        self.posGroups = posGroups if posGroups is not None else set()
        self.negAgents = negAgents if negAgents is not None else set()
        self.category_agents = {i: [] for i in range(self.num_categories)}
        
        if agent_list is None:
            # 15% chance to generate 1-5 tested healthy agents to simulaet use data
            if np.random.rand() < 0.15:
                num_healthy_agents = np.random.choice([1, 2, 3, 4, 5])
                healthy_agents = np.random.choice(range(self.num_agents), size=num_healthy_agents, replace=False)
            else:
                healthy_agents = []

            for agent_id in range(self.num_agents):
                if agent_id in healthy_agents:
                    utility = 0
                    health = 1.0
                else:
                    utility = np.random.choice([1, 2, 3])
                    health = np.random.uniform(0, 1)
                
                # Assign category based on health and utility bins
                utility_bin = int(np.digitize(utility, self.utility_bin_edges) - 1)
                health_bin = int(np.digitize(health, self.health_bin_edges) - 1)
                category = utility_bin * self.health_bins + health_bin
                
                self.category_agents[category].append((agent_id, utility, health))
                self.agents.append((agent_id, utility, health))
        else:
            for agent in agent_list:
                agent_id, utility, health = agent
                utility_bin = np.digitize(utility, self.utility_bin_edges) - 1
                health_bin = np.digitize(health, self.health_bin_edges) - 1
                category = utility_bin * self.health_bins + health_bin
                self.category_agents[category].append((agent_id, utility, health))
                self.agents.append((agent_id, utility, health))
            for category in self.category_agents:
                random.shuffle(self.category_agents[category])
            self.agents.sort(key=lambda x: x[0])
        
        self.selected_agents = set()
        self.category_counts = {i: len(self.category_agents[i]) for i in range(self.num_categories)}
        self.selection_attempts = 0  # Track total selections attempted
        return self._get_state(), {}

    def _get_state(self):
        """Return current state: Counts of remaining agent types, sum of utilities, product of health values."""
        current_utility_sum = sum(self.agents[id][1] for id in self.selected_agents)  # Utility is at index 1
        health_product = np.prod([self.agents[id][2] for id in self.selected_agents]) if self.selected_agents else 1.0  # Health is at index 2
        return np.concatenate([np.array(list(self.category_counts.values()), dtype=np.float32), [current_utility_sum, health_product]])

    def step(self, action, skip_reward=False):
        """Take a step by selecting a category or stopping."""
        done = False
        reward = 0

        if action == self.num_categories:
            done = True
            if not skip_reward:
                reward += self._compute_reward()
        elif self.category_counts[action] > 0:
            while self.category_agents[action]:
                selected_agent = self.category_agents[action].pop()
                self.category_counts[action] -= 1
                
                # Skip agents with utility == 0 or health == 0
                if selected_agent[1] > 0 and selected_agent[2] > 0:
                    self.selected_agents.add(selected_agent[0])
                    break                         
        
        self.selection_attempts += 1
        if self.selection_attempts >= self.max_selection:
            done = True

        return self._get_state(), reward, done, False, {}

    def _compute_reward(self):
        """Compute final reward: Bernoulli on health values, then multiply by sum of utilities."""

        healthStatus = [np.random.binomial(1, agent[2]) for agent in self.agents]
        tests = [self.selected_agents]
        nextAgents = self.selected_agents

        for currentBudget in range(B-1, 0, -1):
            negative = all(healthStatus[agent] == 1 for agent in nextAgents)
            if negative:
                self.negAgents.update(nextAgents)
                newPosGroups = set()
                for posGroup in self.posGroups:
                    newPosGroups.add(posGroup.difference(nextAgents))
                self.posGroups = newPosGroups
                negDict = bayesTheorem(self.agents, posGroups=frozenset(self.posGroups), negAgents=nextAgents)
                testOutcomeAgents = [(idNum, utility, health) for idNum, (utility, health) in negDict.items()]
                fullNext, _ = solveConicSingle(testOutcomeAgents, G)
                nextAgents = {nextAgent[0] for nextAgent in fullNext}
                tests.append(nextAgents)
            else:    
                self.posGroups.add(frozenset(nextAgents.difference(self.negAgents)))
                posDict = bayesTheorem(self.agents, posGroups=frozenset(self.posGroups), negAgents=self.negAgents)
                testOutcomeAgents = [(idNum, utility, health) for idNum, (utility, health) in posDict.items()]
                fullNext, _ = solveConicSingle(testOutcomeAgents, G)
                nextAgents = {nextAgent[0] for nextAgent in fullNext}
                tests.append(nextAgents)

        consideredAgents = set()
        totalUtility = 0
        for currentAgents in tests:
            negative = all(healthStatus[id] == 1 for id in currentAgents)
            if negative:
              totalUtility += sum(self.agents[currentAgent][1] for currentAgent in currentAgents if currentAgent not in consideredAgents)
              consideredAgents = consideredAgents.union(currentAgents)
        return totalUtility

model2 = PPO.load(direct_path_B2)
env2 = AgentSelectionEnvB2()

def use_b2_model(agent_list, posGroups, negAgets, currentB):

    if currentB == 2:
       currentModel = model2
       currentEnv = env2
    
    obs, _ = currentEnv.reset(agent_list, posGroups=posGroups, negAgents=negAgets)
    done = False
    
    while not done:
        action, _ = currentModel.predict(obs)
        obs, _, done, _, _ = currentEnv.step(int(action), skip_reward=True)
    
    return currentEnv.selected_agents

class AgentSelectionEnvB3(AgentSelectionEnvB2):
    def _compute_reward(self):
        """Compute final reward: Bernoulli on health values, then multiply by sum of utilities."""

        healthStatus = [np.random.binomial(1, agent[2]) for agent in self.agents]
        nextAgents = self.selected_agents
        tests = [nextAgents]

        for currentBudget in range(B-1, 0, -1):
            negative = all(healthStatus[agent] == 1 for agent in nextAgents)
            if negative:
                self.negAgents.update(nextAgents)
                newPosGroups = set()
                for posGroup in self.posGroups:
                    newPosGroups.add(posGroup.difference(nextAgents))
                self.posGroups = newPosGroups
                negDict = bayesTheorem(self.agents, posGroups=frozenset(self.posGroups), negAgents=nextAgents)
                testOutcomeAgents = [(idNum, utility, health) for idNum, (utility, health) in negDict.items()]
            else:    
                self.posGroups.add(frozenset(nextAgents.difference(self.negAgents)))
                posDict = bayesTheorem(self.agents, posGroups=frozenset(self.posGroups), negAgents=self.negAgents)
                testOutcomeAgents = [(idNum, utility, health) for idNum, (utility, health) in posDict.items()]

            if currentBudget == 1:
                fullNext, _ = solveConicSingle(testOutcomeAgents, G)
                nextAgents = {nextAgent[0] for nextAgent in fullNext}
                tests.append(nextAgents)
            else:
                nextAgents = use_b2_model(testOutcomeAgents, self.posGroups, self.negAgents, currentBudget)
                tests.append(nextAgents)

        consideredAgents = set()
        totalUtility = 0
        for currentAgents in tests:
            negative = all(healthStatus[id] == 1 for id in currentAgents)
            if negative:
              totalUtility += sum(self.agents[currentAgent][1] for currentAgent in currentAgents if currentAgent not in consideredAgents)
              consideredAgents = consideredAgents.union(currentAgents)

        return totalUtility 

model3 = PPO.load(direct_path_B3)
env3 = AgentSelectionEnvB3()

def use_b3_model(agent_list, posGroups, negAgents, currentB):

    if currentB == 3:
       currentModel = model3
       currentEnv = env3
    
    obs, _ = currentEnv.reset(agent_list, posGroups=posGroups, negAgents= negAgents)
    done = False
    
    while not done:
        action, _ = currentModel.predict(obs)
        obs, _, done, _, _ = currentEnv.step(int(action), skip_reward=True)
    
    return currentEnv.selected_agents

class AgentSelectionEnvB4(AgentSelectionEnvB2):
    def _compute_reward(self):
        """Compute final reward: Bernoulli on health values, then multiply by sum of utilities."""

        healthStatus = [np.random.binomial(1, agent[2]) for agent in self.agents]
        nextAgents = self.selected_agents
        tests = [nextAgents]

        for currentBudget in range(B-1, 0, -1):
            negative = all(healthStatus[agent] == 1 for agent in nextAgents)
            if negative:
                self.negAgents.update(nextAgents)
                newPosGroups = set()
                for posGroup in self.posGroups:
                    newPosGroups.add(posGroup.difference(nextAgents))
                self.posGroups = newPosGroups
                negDict = bayesTheorem(self.agents, posGroups=frozenset(self.posGroups), negAgents=nextAgents)
                testOutcomeAgents = [(idNum, utility, health) for idNum, (utility, health) in negDict.items()]
            else:    
                self.posGroups.add(frozenset(nextAgents.difference(self.negAgents)))
                posDict = bayesTheorem(self.agents, posGroups=frozenset(self.posGroups), negAgents=self.negAgents)
                testOutcomeAgents = [(idNum, utility, health) for idNum, (utility, health) in posDict.items()]

            if currentBudget == 1:
                fullNext, _ = solveConicSingle(testOutcomeAgents, G)
                nextAgents = {nextAgent[0] for nextAgent in fullNext}
                tests.append(nextAgents)
            elif currentBudget == 2:
                nextAgents = use_b2_model(testOutcomeAgents, self.posGroups, self.negAgents, currentBudget)
                tests.append(nextAgents)
            elif currentBudget == 3:
                nextAgents = use_b3_model(testOutcomeAgents, self.posGroups, self.negAgents, currentBudget)
                tests.append(nextAgents)

        consideredAgents = set()
        totalUtility = 0
        for currentAgents in tests:
            negative = all(healthStatus[id] == 1 for id in currentAgents)
            if negative:
              totalUtility += sum(self.agents[currentAgent][1] for currentAgent in currentAgents if currentAgent not in consideredAgents)
              consideredAgents = consideredAgents.union(currentAgents)

        return totalUtility 

model4 = PPO.load(direct_path_B4)
env4 = AgentSelectionEnvB4()

def use_b4_model(agent_list, posGroups, negAgents, currentB):

    if currentB == 4:
       currentModel = model4
       currentEnv = env4
    
    obs, _ = currentEnv.reset(agent_list, posGroups=posGroups, negAgents= negAgents)
    done = False
    
    while not done:
        action, _ = currentModel.predict(obs)
        obs, _, done, _, _ = currentEnv.step(int(action), skip_reward=True)
    
    return currentEnv.selected_agents

class AgentSelectionEnvB5(AgentSelectionEnvB2):
    def reset(self, agent_list=None, posGroups=None, negAgents=None, seed=None, healthStatus=None):
        """Initialize a new episode with a given set of 50 agents or generate a new set."""
        self.agents = []
        self.posGroups = posGroups if posGroups is not None else set()
        self.negAgents = negAgents if negAgents is not None else set()
        self.category_agents = {i: [] for i in range(self.num_categories)}
        
        if agent_list is None:

            for agent_id in range(self.num_agents):
                utility = np.random.choice([1, 2, 3])
                health = np.random.uniform(0, 1)
                
                # Assign category based on health and utility bins
                utility_bin = int(np.digitize(utility, self.utility_bin_edges) - 1)
                health_bin = int(np.digitize(health, self.health_bin_edges) - 1)
                category = utility_bin * self.health_bins + health_bin
                
                self.category_agents[category].append((agent_id, utility, health))
                self.agents.append((agent_id, utility, health))
        else:
            for agent in agent_list:
                agent_id, utility, health = agent
                utility_bin = np.digitize(utility, self.utility_bin_edges) - 1
                health_bin = np.digitize(health, self.health_bin_edges) - 1
                category = utility_bin * self.health_bins + health_bin
                self.category_agents[category].append((agent_id, utility, health))
                self.agents.append((agent_id, utility, health))
            for category in self.category_agents:
                random.shuffle(self.category_agents[category])
            self.agents.sort(key=lambda x: x[0])
        
        self.selected_agents = set()
        self.category_counts = {i: len(self.category_agents[i]) for i in range(self.num_categories)}
        self.selection_attempts = 0  # Track total selections attempted
        self.healthStatus = healthStatus if healthStatus is not None else [np.random.binomial(1, agent[2]) for agent in self.agents]
        return self._get_state(), {}

    def _compute_reward(self):
        """Compute final reward: Bernoulli on health values, then multiply by sum of utilities."""

        nextAgents = self.selected_agents
        tests = [nextAgents]

        for currentBudget in range(B-1, 0, -1):
            negative = all(self.healthStatus[agent] == 1 for agent in nextAgents)
            if negative:
                self.negAgents.update(nextAgents)
                newPosGroups = set()
                for posGroup in self.posGroups:
                    newPosGroups.add(posGroup.difference(nextAgents))
                self.posGroups = newPosGroups
                negDict = bayesTheorem(self.agents, posGroups=frozenset(self.posGroups), negAgents=nextAgents)
                testOutcomeAgents = [(idNum, utility, health) for idNum, (utility, health) in negDict.items()]
            else:    
                self.posGroups.add(frozenset(nextAgents.difference(self.negAgents)))
                posDict = bayesTheorem(self.agents, posGroups=frozenset(self.posGroups), negAgents=self.negAgents)
                testOutcomeAgents = [(idNum, utility, health) for idNum, (utility, health) in posDict.items()]

            if currentBudget == 1:
                fullNext, _ = solveConicSingle(testOutcomeAgents, G)
                nextAgents = {nextAgent[0] for nextAgent in fullNext}
                tests.append(nextAgents)
            elif currentBudget == 2:
                nextAgents = use_b2_model(testOutcomeAgents, self.posGroups, self.negAgents, currentBudget)
                tests.append(nextAgents)
            elif currentBudget == 3:
                nextAgents = use_b3_model(testOutcomeAgents, self.posGroups, self.negAgents, currentBudget)
                tests.append(nextAgents)
            elif currentBudget == 4:
                nextAgents = use_b4_model(testOutcomeAgents, self.posGroups, self.negAgents, currentBudget)
                tests.append(nextAgents)

        consideredAgents = set()
        totalUtility = 0
        for currentAgents in tests:
            negative = all(self.healthStatus[id] == 1 for id in currentAgents)
            if negative:
              totalUtility += sum(self.agents[currentAgent][1] for currentAgent in currentAgents if currentAgent not in consideredAgents)
              consideredAgents = consideredAgents.union(currentAgents)

        return totalUtility 

model5 = PPO.load(direct_path_B5)
env5 = AgentSelectionEnvB5()

def eval_b5_model(agent_list, healthStatus):
    obs, _ = env5.reset(agent_list, healthStatus=healthStatus)
    done = False
    
    while not done:
        action, _ = model5.predict(obs)
        obs, reward, done, _, _ = env5.step(int(action))

    return reward

df = pd.read_csv(file_path)

if column_name not in df.columns:
    df[column_name] = None

# Process each row with a granular progress bar
batch_size = 100
for i in range(len(df)):
    # Process the current row
    agent_data = ast.literal_eval(df.at[i, 'agents'])
    healthStatus = ast.literal_eval(df.at[i, 'healthStatus'])
    df.at[i, column_name] = eval_b5_model(agent_data, healthStatus)
    # df.at[i, function.__name__] = useSeqSample(agent_data)
    
    # Save after every 100 rows
    if i > 0 and i % batch_size == 0:
        df.to_csv(file_path, index=False)

# Save again at the end to ensure the last part is saved
df.to_csv(file_path, index=False)