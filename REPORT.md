# Implementation



# Learning Algorithm
Learning Algorithm: Q-Learning
Hyperparameters:
Neural Network Architecture:


# Hyperparameters
BATCH_SIZE = 64         # minibatch size

GAMMA = 0.99            # discount factor

TAU = 1e-3              # for soft update of target parameters

LR = 5e-4               # learning rate 

UPDATE_EVERY = 4        # how often to update the network




n_episodes=1000         # maximum episodes to take 

max_t=1000              # maximum steps to take in one episodes

eps_start=1.0           # start with epsilon 1 to take random actions at first

eps_end=0.01            # minimum episolon to keep

eps_decay=0.995         # decaying rate of epsilon


# Score 

The score with +15 reward over the last 100 consecutive tasks is in the "checkpoints" folder as "score.png".


# Idea for Future Work
Impement the rest of the Rainbow algorithm, such as Double Q-Network, prioritized experience replay, etc.



