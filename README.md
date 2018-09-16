# Overview

Part of the Deep Reinforcement Learning Nano Degree(drlnd) from Udacity.

# Installation


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



