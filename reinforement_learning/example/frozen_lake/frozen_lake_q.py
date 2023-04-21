import gymnasium as gym
import random
import numpy as np


environment = gym.make("FrozenLake-v1", is_slippery=True)
environment.reset()

import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 17})

# We re-initialize the Q-table
qtable = np.zeros((environment.observation_space.n, environment.action_space.n))

# Hyperparameters
episodes = 20000        # Total number of episodes
alpha = 0.5            # Learning rate
gamma = 0.9            # Discount factor
epsilon = 0.01          # epsilon-greedy 

# List of outcomes to plot
outcomes = []

print('Q-table before training:')
print(qtable)

# Training
for _ in range(episodes):
    state = environment.reset()[0]
    terminated = False
    truncated =False
    done = truncated|terminated 
    # By default, we consider our outcome to be a failure
    outcomes.append("Failure")

    # Until the agent gets stuck in a hole or reaches the goal, keep training it
    while not done:
        # Choose the action with the highest value in the current state
        if np.random.random()<epsilon:
            action = environment.action_space.sample()
        else:
            if np.max(qtable[state]) > 0:
                action = np.argmax(qtable[state])

        # If there's no best action (only zeros), take a random one
            else:
                action = environment.action_space.sample()
             
        # Implement this action and move the agent in the desired direction
        new_state, reward,terminated,truncated, info = environment.step(action)
        done = truncated|terminated 
        # Update Q(s,a)
        qtable[state, action] = qtable[state, action] + \
                                alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])
        
        # Update our current state
        state = new_state

        # If we have a reward, it means that our outcome is a success
        if reward:
          outcomes[-1] = "Success"

print()
print('===========================================')
print('Q-table after training:')
print(qtable)

# Plot outcomes

# plt.figure(figsize=(12, 5))
# plt.xlabel("Run number")
# plt.ylabel("Outcome")
# ax = plt.gca()
# ax.set_facecolor('#efeeea')
# plt.bar(range(len(outcomes)), outcomes, color="#0A047A", width=1.0)
# plt.show()


# test
nb_success=0
episodes=2000

for i in range(episodes):
    state = environment.reset()[0]
    terminated = False
    truncated =False
    done = truncated|terminated 
    while not done:
        if np.max(qtable[state]) > 0:
            action = np.argmax(qtable[state])
        # If there's no best action (only zeros), take a random one
        else:
            action = environment.action_space.sample()
        new_state, reward,terminated,truncated, info = environment.step(action)
        state = new_state
        done = truncated|terminated 
        nb_success+=reward
print (f"Success rate = {nb_success/episodes*100}%")