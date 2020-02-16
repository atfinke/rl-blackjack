

import gym
import sys
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

from source import BlackjackEnv
import numpy as np

sys.path.insert(1, './')
sys.path.append('/Users/andrewfinke/opt/miniconda3/envs/rl-blackjack/lib/python3.8/site-packages')
env = gym.make('BlackjackEnv-v0')
state = env.reset()

num_of_available_states = env.observation_space.n
num_of_available_actions = env.action_space.n

q = np.zeros(shape=(num_of_available_states, num_of_available_actions))
n = np.zeros(shape=(num_of_available_states, num_of_available_actions))

steps = 100_000
state = env.reset()
rewards = np.zeros(steps)
