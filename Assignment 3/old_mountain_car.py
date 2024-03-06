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
bins = np.array([(100, 100)] * 10)
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


def QAC(steps=10000, gamma=0.95, alpha=0.5, beta=0.5):
    env = gym.make('MountainCar-v0', render_mode='human')
    n_features = features_table.tiling_size ** 2 * features_table.layers * features_table.action_size
    theta = np.ones(n_features)
    w = np.ones(n_features)

    state, info = env.reset()
    action = sample_action(state, theta)
    rewards = []
    curr_accumulated_reward = 0
    for i in tqdm(range(steps), desc=f'QAC: r: {curr_accumulated_reward}'):
        next_state, reward, terminated, truncated, info = env.step(action)
        curr_accumulated_reward += reward
        next_action = sample_action(next_state, theta)

        td_error = reward + gamma * get_q_value(next_state, next_action, w) - get_q_value(state, action, w)
        grad_policy = features_table.get_features_vector(state, action) - np.mean(
            [features_table.get_features_vector(state, a) for a in ALL_ACTIONS])
        theta += alpha * grad_policy * get_q_value(state, action, w)
        w += beta * td_error * features_table.get_features_vector(state, action)

        action = next_action
        state = next_state

        if terminated or truncated:
            state, info = env.reset()
            action = sample_action(state, theta)
            rewards.append(curr_accumulated_reward)
            curr_accumulated_reward = 0

    env.close()
    return w, theta, rewards
