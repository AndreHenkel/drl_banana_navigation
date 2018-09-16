import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """ QNetwork which implements the Dueling Architecture.
        The Dueling Architecture is a double stream FeedForward Neural Network, 
        that predicts the value and the advantage of a situation.
        Since the value often doesn't change alot after some training, 
        it is useful to converge better with this kind of architecture
        in comparison with the standard single stream QNetwork.
    """

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        
        
        self.fc2_val = nn.Linear(fc1_units, fc2_units)
        self.fc2_adv = nn.Linear(fc1_units, fc2_units)
        
        self.fc3_val = nn.Linear(fc2_units, 1)
        self.fc3_adv = nn.Linear(fc2_units, action_size)
        
        self.action_size = action_size

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        
        val = F.relu(self.fc2_val(x))
        adv = F.relu(self.fc2_adv(x))
        
        # format val to calculate together with adv
        val = self.fc3_val(val).expand(x.size(0), self.action_size)
        adv = self.fc3_adv(adv)
        
        # calculate the action by taking the value and the advantage
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.action_size)
        return x
