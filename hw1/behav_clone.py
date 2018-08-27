import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

"""
Code to load an expert policy and generate a behavioral cloning model.

Example usage:
    python behav_clone.py expert_data/Ant-v2.pkl Ant-2.pkl --param xxx

Author of this script: Nathan Lambert nol@berkeley.edu
"""
