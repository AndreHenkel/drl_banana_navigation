# Overview

Part of the Deep Reinforcement Learning Nano Degree(drlnd) from Udacity.
In this project you needed to train an Agent to collect yellow bananas in an Unity Environment whilst avoiding blue bananas.
The agent sees an 37 dimensional state and an action space of 4.

The basic structure of this project was taken from the examples of the Udacity DRLND and changed slightly to fit the new criteria.

# Setup

To start, you need Python 3.6, PyTorch, Numpy, Matplotlib, and the Unity ML-Agents Toolkit.

With Anaconda or virtualenv you can create your python environment like:
conda create -n drlnd python=3.6 pytorch matplotlib

For the Unity ML-Agent you need to download the Toolkit (https://github.com/Unity-Technologies/ml-agents) go to the ml-agents/python directory and install it via:

pip install ml-agents/python

# Instructions

Watch the trained agent with:
$ python watch.py

and let the agent train again with:
$ python train.py


# Environment

Unity's Banana Environment.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

    0 - move forward.
    1 - move backward.
    2 - turn left.
    3 - turn right.

The task is episodic, and the agent solvedthe environment, after getting an average score of +15 over 100 consecutive episodes.


# Architecture

A Dueling Q-Network to predict the value and advantage differently in two streams.


# Algorithms

Fixed Target Q-Network
Experience Replay
Dueling Q-Network architecture


# Future

For the future additonal algorithms can be implemented to then end up as the Rainbow Algorithm, 
which combines a number of different techniques to improve the Algorithm.



