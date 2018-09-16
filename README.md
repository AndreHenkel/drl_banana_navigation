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
The state space is 37 in which the agent sees a ray perception of it's environment in fron of him.
The action space is 4, for the movements: 'forward', 'left', 'right', 'backwards'
The environment is considered solved after 100 consecutive episodes with an average score over +15.


# Architecture

A Dueling Q-Network to predict the value and advantage differently in two streams.


# Algorithms

Fixed Target Q-Network
Experience Replay
Dueling Q-Network architecture


# Future

For the future additonal algorithms can be implemented to then end up as the Rainbow Algorithm, 
which combines a number of different techniques to improve the Algorithm.



