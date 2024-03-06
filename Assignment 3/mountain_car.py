import gymnasium as gym
import numpy as np
from tqdm import tqdm


class TileCoding:
    # self.position_offsets = np.array([-0.1, 0, 0.1, 0.2])
    # self.velocity_offsets = np.array([-0.01, 0, 0.01, 0.02])
    # self.num_layers = 4
    # self.num_tiles = 5

    def __init__(self):
        self.position_min, self.position_max = -1.2, 0.6
        self.velocity_min, self.velocity_max = -0.07, 0.07
        self.position_offsets = np.array([-0.2, -0.1, 0, 0.1, 0.2, 0.3])
        self.velocity_offsets = np.array([-0.02, -0.01, 0, 0.01, 0.02, 0.03])
        self.num_layers = 6
        self.num_tiles = 4
        self.layers_bins = []
        self.init_tilings()

    def create_layer(self, position_offset=0.0, velocity_offset=0.0):
        position_bins = np.linspace(self.position_min + position_offset, self.position_max + position_offset,
                                    num=self.num_tiles + 1,
                                    endpoint=True)
        velocity_bins = np.linspace(self.velocity_min + velocity_offset, self.velocity_max + velocity_offset,
                                    num=self.num_tiles + 1,
                                    endpoint=True)
        layer = (position_bins, velocity_bins)
        return layer

    def init_tilings(self):
        for i in range(self.num_layers):
            self.layers_bins.append(self.create_layer(self.position_offsets[i], self.velocity_offsets[i]))

    def tile_encode(self, position, velocity):
        encoded_vec = np.zeros((self.num_tiles ** 2 * self.num_layers, 1))
        for i, (position_bins, velocity_bins) in enumerate(self.layers_bins):
            position_i = np.digitize(position, position_bins) - 1
            velocity_i = np.digitize(velocity, velocity_bins) - 1
            encoded_vec[position_i + velocity_i * self.num_tiles + i * self.num_tiles ** 2] = 1
        return encoded_vec


def get_action(tile_coding, state, theta):
    features = tile_coding.tile_encode(state[0], state[1])
    action_probabilities = softmax(theta.T @ features)
    return np.random.choice(action_probabilities.shape[0], p=action_probabilities.flatten())


def get_q_value(tile_coding, state, w):
    features = tile_coding.tile_encode(state[0], state[1])
    return (w.T @ features).flatten()


def softmax(vec):
    pi = np.exp(vec - max(vec))
    pi = pi / pi.sum()
    return pi.reshape(-1, 1)


def get_expected_feature(tile_coding, state, pi):
    features = tile_coding.tile_encode(state[0], state[1])
    return features @ pi.T


def estimate_policy(env, tile_coding, theta, steps=1000):
    episodes_counter = 0
    i = 0
    while i < steps:
        state, info = env.reset()
        while True:  # begin episode
            action = get_action(tile_coding, state, theta)
            state, reward, terminated, truncated, info = env.step(action)
            i += 1
            if terminated or truncated:
                break
        episodes_counter += 1
    return episodes_counter


def QAC(steps=10000, gamma=0.95, alpha=0.02, beta=0.08):
    print("Running QAC...")
    env = gym.make('MountainCar-v0')

    tile_coding = TileCoding()
    num_features = tile_coding.num_tiles ** 2 * tile_coding.num_layers
    w = np.ones((num_features, 1))
    theta = np.ones((num_features, env.action_space.n))

    policy_estimation = []

    state, info = env.reset()
    for i in tqdm(range(int(steps)), desc="Steps"):

        action = get_action(tile_coding, state, theta)
        next_state, reward, terminated, truncated, info = env.step(action)

        delta = reward + gamma * get_q_value(tile_coding, next_state, w) - get_q_value(tile_coding, state, w)
        w += beta * delta * tile_coding.tile_encode(state[0], state[1])

        features = tile_coding.tile_encode(state[0], state[1])
        pi = softmax(theta.T @ features)
        features_mat = np.zeros((num_features, env.action_space.n))
        features_mat[:, action] = features.flatten()
        theta += alpha * delta * (features_mat - get_expected_feature(tile_coding, state, pi))

        if i % 100 == 0:
            policy_estimation.append(estimate_policy(env, tile_coding, theta))

        state = next_state
        if terminated or truncated:
            state, info = env.reset()

    return w, theta, policy_estimation


def play(theta):
    env = gym.make('MountainCar-v0', render_mode='human')
    tile_coding = TileCoding()
    state, info = env.reset()
    while True:
        action = get_action(tile_coding, state, theta)
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.close()
