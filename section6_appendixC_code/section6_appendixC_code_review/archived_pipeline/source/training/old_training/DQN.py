import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym import spaces
from stable_baselines3 import DQN

class ItemSelectionEnv(gym.Env):
    def __init__(self, num_items=50, max_selection=5):
        super(ItemSelectionEnv, self).__init__()
        self.num_items = num_items
        self.max_selection = max_selection
        
        # Observation space: (Current Utility Sum, Product of Selected Health Values)
        self.observation_space = spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32)
        
        # Action space: Select an item (0 to 49) or stop (50)
        self.action_space = spaces.Discrete(num_items + 1)
        
        self.reset()
    
    def reset(self):
        """Initialize a new episode."""
        self.health_values = np.random.uniform(0, 1, self.num_items)
        self.utility_values = np.random.choice([0, 1, 2, 3], self.num_items, p=[0.25, 0.25, 0.25, 0.25])
        self.selected_items = []
        return self._get_state()
    
    def _get_state(self):
        """Return state: (Current Utility Sum, Product of Selected Health Values)."""
        current_utility_sum = sum(self.utility_values[i] for i in self.selected_items)
        health_product = np.prod([self.health_values[i] for i in self.selected_items]) if self.selected_items else 1.0
        return np.array([current_utility_sum, health_product], dtype=np.float32)
    
    def step(self, action):
        """Take a step in the environment."""
        done = False
        reward = 0
        
        if action == self.num_items or len(self.selected_items) >= self.max_selection:
            done = True
            reward = self._compute_reward()
        elif action in self.selected_items:
            # Prevent duplicate selections with a small penalty
            return self._get_state(), -1, False, {}
        else:
            self.selected_items.append(action)
        
        return self._get_state(), reward, done, {}
    
    def _compute_reward(self):
        """Compute final reward based on utility sum."""
        return sum(self.utility_values[i] for i in self.selected_items)
    
    def render(self):
        print(f"Selected items: {self.selected_items}, Total Utility: {self._compute_reward()}")

# Create environment
env = ItemSelectionEnv()

# Create DQN model
model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.001, buffer_size=50000, batch_size=32, gamma=0.99, exploration_fraction=0.1)

# Train the model
print("Training DQN model...")
model.learn(total_timesteps=100000)

# Save model
model.save("dqn_item_selection")
print("Model saved as dqn_item_selection")

# Load and test trained model
model = DQN.load("dqn_item_selection")

def select_items(model, health_values, utility_values):
    """Runs the trained model on a specific instance of items."""
    num_items = len(health_values)
    env = ItemSelectionEnv(num_items=num_items, max_selection=5)
    
    # Manually set the environment's health and utility values
    env.health_values = np.array(health_values)
    env.utility_values = np.array(utility_values)
    env.selected_items = []
    
    obs = env._get_state()
    done = False

    print("Selecting items...")
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        env.render()
    
    return env.selected_items, reward

# Example usage with real data
health_values = np.random.uniform(0, 1, 50)
utility_values = np.random.choice([0, 1, 2, 3], 50, p=[0.25, 0.25, 0.25, 0.25])

selected_items, final_reward = select_items(model, health_values, utility_values)
print(f"Selected Items: {selected_items}")
print(f"Final Reward: {final_reward}")
