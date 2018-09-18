# Implementation

For our Agent to actually solve the task we implemented the famous Deep Q-Learning(DQN).
Since Agents with DQN are likely to overestimate or get carried away easily, additions to the standard algorithm has been made.
Namely Experience replay to store the experienced data and learn from it again, similar as the human brain does by recapturing experiences like when you sleep.
Additionally Fixed Q targets have been implemented, to not directly use the weights from the latest episode/backprop to calculate the new action,
as this could have errors that build up over time.


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


# Neural Network Architecture
For the Neural Network,
we implemented the Dueling Architecture with a first hidden layer of 64 units and then separating to two streams.
Both stream have also 64 units as their hidden layer. 
One stream predicts the value of the current situation and the other estimates the adventage. 
At the end, both streams are put together to predicct the next action. 

As opposed to the normal DQN, where only one stream is used.

# Score 

The score with +15 reward over the last 100 consecutive tasks is in the "checkpoints" folder as "score.png".


# Idea for Future Work
Impement the rest of the Rainbow algorithm, such as Double Q-Network, prioritized experience replay, etc.



