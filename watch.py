from unityagents import UnityEnvironment
import numpy as np

from collections import namedtuple, deque
import time

import torch

from drl.agent import Agent
from drl.model import QNetwork


seed = 1337
n_episodes = 10
agent_slowing = 0.1

#get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# get UnityEnvironment
env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")
print("Started UnityEnvironment")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=False)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
state_size = len(state)
print('States have length:', state_size)


model = QNetwork(state_size, action_size, seed).to(device)
model.load_state_dict(torch.load('checkpoints/checkpoint_dueling_dqn.pth'))

scores = []                         # list containing scores from each episode
scores_window = deque(maxlen=3)     # last 3 scores                  # initialize epsilon

# activate eval and no_grad mode to just watch the agent
model.eval()
torch.no_grad()
for i_episode in range(1, n_episodes+1):
    env_info = env.reset(train_mode=False)[brain_name] # reset the environment
    state = env_info.vector_observations[0] 
    score = 0
    steps_done=0
    for t in range(1000):

        # lets the agent react slower for better visualization for the human watcher
        time.sleep(agent_slowing)
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        action_values = model(state)
        action = np.argmax(action_values.cpu().data.numpy())
        
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        
        score += reward                                # update the score
        state = next_state                             # roll over the state to next time step
        steps_done=steps_done+1
        print(steps_done)
        if done:                                       # exit loop if episode finished
            break
    print(steps_done)    
    scores_window.append(score)       # save most recent score
    scores.append(score)              # save most recent score
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

env.close()
