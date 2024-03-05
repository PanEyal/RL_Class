import gymnasium as gym
import numpy as np
from tqdm import tqdm

import tile_coding
import importlib
importlib.reload(tile_coding)
from tile_coding import *

ALL_ACTIONS = np.arange(3)
low = [-1.2, -0.07]
high = [0.6, 0.07]
bins = np.array([(100, 100)] * 5)
offsets = np.array([(-0.1, -0.01), (0, 0), (0.1, 0.01), (0.2, 0.02)])
features_table = TiledFeaturesTable(low, high, bins, offsets, action_size=len(ALL_ACTIONS))


def sample_action(state, theta):
    feature_vecs = [features_table.get_features_vector(state, a) for a in ALL_ACTIONS]
    feature_times_theta = [vec @ theta for vec in feature_vecs]
    distribution = softmax(np.array(feature_times_theta))
    action = np.random.choice(ALL_ACTIONS, p=distribution)
    return action


def softmax(d):
    exp_d = np.exp(d - np.max(d))
    assert not np.any(np.isnan(exp_d / np.sum(exp_d))), f'exp_d: {exp_d}, d:{d}'
    return exp_d / exp_d.sum()


def get_q_value(state, action, w):
    feature_vec = features_table.get_features_vector(state, action)
    return feature_vec @ w

a = create_tiling_grid([-1.2, -0.07], [0.6, 0.07], (4, 4), (0, 0))
print(a)