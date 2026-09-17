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

        # Pooled testing size constraint

        M.constraint("pool", Expr.sum(x), Domain.lessThan(G))

        M.solve()

        # print(agents)
        # print(tuple(agent for agent, level in zip(agents, np.round_(x.level())) if level == 1))
        # print(M.primalObjValue())

        utility = math.exp(M.primalObjValue())
        strategy = tuple(agent for agent, level in zip(agents, np.round_(x.level())) if level == 1)

        return strategy, utility
    
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

# Define the number of agents and the group test parameters
N = 50  
B = 2 # TODO 
layers = 7 # TODO 
num_samples = 100000
dropout = "none" # TODO
health_buckets = 4
utility_buckets = 3
save_path = 'seq_types/'
G = 5
d = 3
input_size = utility_buckets * health_buckets
output_size = N  

def generate_classes(health_buckets, utility_buckets):
    health_step = 1.0 / health_buckets
    classes = []

    for h in range(health_buckets):
        health_lower = h * health_step
        health_upper = (h + 1) * health_step
        
        for u in range(utility_buckets):
            group = {
                "health_lower": health_lower,
                "health_upper": health_upper,
                "utility_lower": u,
                "utility_upper": u + 1
            }
            classes.append(group)
            print(f"{group}\n")

    return classes

classOrder = generate_classes(health_buckets, utility_buckets)
print(classOrder)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(processName)s - %(message)s",
    handlers=[
        logging.FileHandler(f"Seq_G{B}_{layers}_dropout{dropout}.log"),
        logging.StreamHandler()
    ]
)

def sample_actions(probabilities, agents, classOrder=classOrder, G=G):
    
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.detach().cpu()  # Ensure it's on CPU for sampling
    else:
        probabilities = torch.tensor(probabilities, dtype=torch.float32)  # Convert to tensor

    valid_count = (probabilities > 0).sum().item()
    group_size = min(G, valid_count)

    selected_indices = torch.multinomial(probabilities, group_size, replacement=False)


    return selected

def use_RL_model(model, agents):

    input_data = preprocess_agents(agents)
    input_data = input_data.unsqueeze(0)  # Add batch dimension
    predicted_probabilities = model(input_data)

    # Make predictions (forward pass)
    with torch.no_grad():  # Disable gradient calculation for inference
        predicted_probabilities = model(input_data)

    predicted_bool_list = sample_actions(predicted_probabilities[0]).tolist()

    return predicted_bool_list

def load_RL_model_if_exists(model, samples, epoch, beta, beta_decay, lr, save_path=save_path):
    model_filename = f"{save_path}dynamic_RL_model_N{N}_Seq_G{G}_B{B}_s{samples}_e{epoch}_b{beta}_lr{lr}_bd{beta_decay}_dropout{dropout}.pth"
    if os.path.exists(model_filename):
        model.load_state_dict(torch.load(model_filename))
        print(f"Loaded model from {model_filename}", flush = True)
        logging.info(f"Loaded model from {model_filename}")
        return True
    return False

#TODO
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
        
        x = F.softmax(self.fc_out(x), dim=-1) 
        return x

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

model2 = SeqRLModel(input_size=input_size, output_size=output_size)
load_RL_model_if_exists(model2, num_samples, 10, 0, 0, 0.1, newB=2) # TODO: params

def nextAction(agents, newB):
   if newB >= 2:
      # predicted_probabilities = modelB2(input_data)
      # predicted_probabilities = torch.clamp(predicted_probabilities, 0.0, 1.0)

      # Sample actions based on the predicted probabilities
      # print(f"{time.time()-now}, sample", flush=True)
      # now = time.time()
      if newB == 2:
        m = use_RL_model(modelB2, agents, newB)
      if newB == 3:
        m = use_RL_model(modelB3, agents, newB)

      # Convert sampled boolean list to a tree structure and compute utility (reward)

      selection = frozenset(np.where(m)[0])

      pNegative = prod(agent[2] for agent in agents if agent[0] in selection)

      # print(f"{time.time()-now}, iters", flush=True)
      # now = time.time()
      if pNegative < 1:
        ########## todo for larger B, not use bayes, use gibbs mcmc for both. apply changes to eval
        # print(f"{time.time()-now}, bayes", flush=True)
        # now = time.time()
        posDict = bayesTheorem(agents, posGroups={selection}, negAgents={})
        posOutcomeAgents = sorted(((idNum, utility, health) for idNum, (utility, health) in posDict.items()), key=lambda x: x[0])
        # print(f"{time.time()-now}, next", flush=True)
        # now = time.time()
        posAction = nextAction(posOutcomeAgents, newB-1)
      else:
          posAction = [False] * N

      if pNegative > 0:
        #### TODO for larger B, need to add a fix -- model should not want to include neg people in more tests, waste of space for G.
        # print(f"{time.time()-now}, b2", flush=True) 
        # now = time.time()
        negDict = bayesTheorem(agents, posGroups={}, negAgents=selection)
        negOutcomeAgents = sorted(((idNum, utility, health) for idNum, (utility, health) in negDict.items()), key=lambda x: x[0])
        # process here, if tested and negative
        # print(f"{time.time()-now}, next", flush=True)
        # now = time.time()
        negAction = nextAction(negOutcomeAgents, newB-1)
      else:
          negAction = [False] * N

      return m.tolist() + posAction + negAction
      
   if newB == 1:
      ids, _ = singleSameUtility(agents, G)
      # action = {}
      # Create the boolean list
      selection = [i in ids for i in range(N)]
      return selection

# Training loop using REINFORCE with model saving
def train_RL_model(model, optimizer, training_data, num_epochs, lr, beta, beta_decay, save_interval=100, save_path=save_path):

    # Function to save the model
    def save_RL_model(model, samples, epoch, save_path):
        model_filename = f"{save_path}dynamic_RL_model_N{N}_Seq_G{G}_B{B}_s{samples}_e{epoch}_b{beta}_lr{lr}_bd{beta_decay}_dropout{dropout}.pth"
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
    # Now continue training from start_epoch to num_epochs
    for epoch in tqdm(range(start_epoch, num_epochs)):
        epoch_loss = 0
        epoch_reward = 0
        current_beta = beta * ((1-beta_decay) ** epoch)

        epoch_time = time.time()
        print(f"begin epoch {epoch}", flush=True)

        # print(f"Current beta: {current_beta}")

        start = time.time()
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

            B_tensor = torch.tensor([B], dtype=torch.float32, device=input_data.device).unsqueeze(0)  # Shape: (1, 1)
            input_data = torch.cat((B_tensor, input_data), dim=-1) 
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

            selection = frozenset(np.where(m)[0])

            pNegative = prod(agent[2] for agent in agents if agent[0] in selection)

            # print(f"{time.time()-now}, iters", flush=True)
            # now = time.time()
            if pNegative < 1:
              ########## todo for larger B, not use bayes, use gibbs mcmc for both. apply changes to eval
              # print(f"{time.time()-now}, bayes", flush=True)
              # now = time.time()
              posDict = bayesTheorem(agents, posGroups={selection}, negAgents={})
              posOutcomeAgents = sorted(((idNum, utility, health) for idNum, (utility, health) in posDict.items()), key=lambda x: x[0])
              # print(f"{time.time()-now}, next", flush=True)
              # now = time.time()
              posAction = nextAction(posOutcomeAgents, B-1)
            else:
               posAction = [False] * N

            if pNegative > 0:
              #### TODO for larger B, need to add a fix -- model should not want to include neg people in more tests, waste of space for G.
              # print(f"{time.time()-now}, b2", flush=True) 
              # now = time.time()
              negDict = bayesTheorem(agents, posGroups={}, negAgents=selection)
              negOutcomeAgents = sorted(((idNum, utility, health) for idNum, (utility, health) in negDict.items()), key=lambda x: x[0])
              # process here, if tested and negative
              # print(f"{time.time()-now}, next", flush=True)
              # now = time.time()
              negAction = nextAction(negOutcomeAgents, B-1)
            else:
               negAction = [False] * N

            # print(f"{time.time()-now}, tree", flush=True)
            # now = time.time()
            boolean_tree = boolean_list_to_tree(m.tolist() + posAction + negAction, len(agents))
            # print(f"{time.time()-now}, analyze", flush=True)
            # now = time.time()
            reward = analyzeTree(boolean_tree, agents)
            # print(f"Boolean tree: {boolean_tree}")
            # print(f"{time.time()-now}, opt", flush=True)
            # now = time.time()

            # Policy gradient loss (-log_prob * reward)
            log_probs_actions = m.float() * torch.log(predicted_probabilities + 1e-8) + (1 - m.float()) * torch.log(1 - predicted_probabilities + 1e-8)

            log_probs = log_probs_actions.sum(dim=1)  # Sum across the boolean list
            # print(f"Log probs: {log_probs}")

            # Compute entropy
            entropy = -torch.sum(predicted_probabilities * log_probs)

            # print(f"Entropy: {entropy}")
            # print(f"Reward {reward}")

            # Policy gradient loss (-log_prob * reward) 
            loss = -(log_probs * reward) + -(current_beta * entropy)

            loss = loss.mean()

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

            # print(f"{time.time()-now}, iters", flush=True)
            # now = time.time()

            actions = [selection]

            negative = all(bern_samples[[i for i, agent in enumerate(agents) if agent[0] in selection]] == 1)

            for newB in range(B-1, 0, -1):

              # print(f"neg: {negative}", flush=True)
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
                selection = nextAction(negOutcomeAgents, newB)
                actions.append(selection)
              else:
                ########## todo for larger B, not use bayes, use gibbs mcmc for both. apply changes to eval
                # print(f"{time.time()-now}, bayes", flush=True)
                # now = time.time()
                posDict = bayesTheorem(agents, posGroups={selection}, negAgents={})
                posOutcomeAgents = sorted(((idNum, utility, health) for idNum, (utility, health) in posDict.items()), key=lambda x: x[0])
                # print(f"{time.time()-now}, next", flush=True)
                # now = time.time()
                selection = nextAction(posOutcomeAgents, newB)
                actions.append(selection)
              
              negative = all(bern_samples[[i for i, agent in enumerate(agents) if agent[0] in selection]] == 1)

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
            # print(actions, flush=True)
            # print(reward, flush=True)
            # print(agents, flush=True)
            # print(xys)
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
    return {"lr": lr, "beta": beta, "num_epochs": num_epochs, "beta_decay": beta_decay, "avg_reward": avg_reward}

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


