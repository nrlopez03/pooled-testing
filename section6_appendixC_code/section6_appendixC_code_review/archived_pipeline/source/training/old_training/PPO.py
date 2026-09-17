import gym
import numpy as np
from gym import spaces
from stable_baselines3 import PPO

class ItemSelectionEnv(gym.Env):
    def __init__(self, num_items=50, max_selection=5):
        super(ItemSelectionEnv, self).__init__()
        self.num_items = num_items  # Total items
        self.max_selection = max_selection  # Max items selectable

        # Observation space: health values + utility values + binary indicators for selection
        self.observation_space = spaces.Box(low=0, high=1, shape=(2 * num_items + num_items,), dtype=np.float32)
        
        # Action space: Select one of 50 items or stop
        self.action_space = spaces.Discrete(num_items)
        
        self.reset()

    def reset(self):
        """Initialize a new episode with random item attributes."""
        self.health_values = np.random.uniform(0, 1, self.num_items)
        self.utility_values = np.random.choice([0, 1, 2, 3], self.num_items, p=[0.25, 0.25, 0.25, 0.25])
        
        self.selected = np.zeros(self.num_items)  # No items selected initially
        self.selected_items = []
        return self._get_state()
    
    def _get_state(self):
        """Return the state: Health + Utility + Selection Status."""
        return np.concatenate([self.health_values, self.utility_values / 3.0, self.selected])
    
    def step(self, action):
        """Take a step in the environment."""
        done = False
        reward = 0
        
        if len(self.selected_items) >= self.max_selection:
            # "Stop" action or max selections reached
            done = True
            reward = self._compute_reward()
        else:
            if action not in self.selected_items:
                self.selected_items.append(action)
                self.selected[action] = 1  # Mark item as selected
        
        return self._get_state(), reward, done, {}
    
    def _compute_reward(self):
        """Compute the final reward: sum of selected item utilities."""
        return sum(self.utility_values[i] for i in self.selected_items)
    
    def render(self):
        print(f"Selected items: {self.selected_items}, Total Utility: {self._compute_reward()}")

# Create environment
env = ItemSelectionEnv()

# Create PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
print("Training PPO model...")
model.learn(total_timesteps=10000)

# Save model
model.save("ppo_item_selection")
print("Model saved as ppo_item_selection")

# Load and test trained model
model = PPO.load("ppo_item_selection")
obs = env.reset()
done = False

print("Testing trained model...")
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    env.render()

print(f"Final reward: {reward}")

from stable_baselines3 import PPO

# Load the trained PPO model
model = PPO.load("ppo_item_selection")

import numpy as np

def select_items(model, health_values, utility_values):
    """Runs the trained model on a specific instance of items."""
    
    num_items = len(health_values)  # Ensure correct item count
    env = ItemSelectionEnv(num_items=num_items, max_selection=5)  # Create environment
    
    # Manually set the environment's health and utility values
    env.health_values = np.array(health_values)
    env.utility_values = np.array(utility_values)
    env.selected = np.zeros(num_items)  # Reset selected items
    env.selected_items = []

    obs = env._get_state()
    done = False

    print("Selecting items...")
    while not done:
        action, _ = model.predict(obs)  # Get action from trained model
        obs, reward, done, _ = env.step(action)
        env.render()  # Show selected items

    return env.selected_items, reward  # Return selected items and final reward

# Define a specific instance of items
health_values = np.random.uniform(0, 1, 50)  # Example health values (0 to 1)
utility_values = np.random.choice([0, 1, 2, 3], 50, p=[0.25, 0.25, 0.25, 0.25])  # Example utility values

# Use the trained model to select items
selected_items, final_reward = select_items(model, health_values, utility_values)

print(f"Selected Items: {selected_items}")
print(f"Final Reward: {final_reward}")

