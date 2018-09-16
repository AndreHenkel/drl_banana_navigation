# imports
from unityagents import UnityEnvironment
import numpy as np

import time # for sleep to watch agent better
import matplotlib.pyplot as plt
#%matplotlib inline

from drl.agent import Agent
from drl.dqn import DQN

seed=1337
min_score=15.0

# get UnityEnvironment
env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")
print("Started UnityEnvironment")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
state_size = len(state)
print('States have length:', state_size)

# init agent 
agent = Agent(state_size=state_size, action_size=action_size, seed=seed)

# play
scores = DQN(env, agent, brain_name, min_score)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
fig.savefig('checkpoints/score.png')

env.close()
