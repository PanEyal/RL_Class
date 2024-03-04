import numpy as np
import gymnasium as gym

# Here we will implement the policy iteration algorithm for the frozen_lake environment.

# State representation: Integer from 0 to 15
ALL_STATES = list(range(16))
# Action representation: 0 -> left, 1 -> down, 2 -> right, 3 -> up
ALL_ACTIONS = list(range(4))

ITERATION_LIMIT = 10


def play_episode(env, pi):
    observation, info = env.reset()
    episode = []
    while True:
        action = pi[observation]
        next_observation, reward, terminated, truncated, info = env.step(action)
        episode.append((observation, reward))
        if terminated or truncated:
            break
        observation = next_observation

    return episode


def first_visit_monte_carlo_policy_evaluation(env, pi, episodes=1000, gamma=1.0):
    n_start = 0
    s_start = 0

    for _ in range(episodes):
        episode = play_episode(env, pi)
        g = 0
        for (state, reward) in reversed(episode):
            g = gamma * g + reward
        n_start += 1
        s_start += g

    return s_start / n_start


def q_learning(env, q, episodes=1000, alpha=0.1, gamma=1.0, index_of_episode=0):
    # q: Dict(state -> Dict(action -> value))
    if q is None:
        q = dict(zip(ALL_STATES,
                     [dict(zip(ALL_ACTIONS,
                               [0. for _ in ALL_ACTIONS]))
                      for _ in ALL_STATES]))

    for i in range(episodes):
        state, info = env.reset()
        while True:
            # choose action with e-greedy policy
            action_probs = get_best_action_probs(q[state], 1 / (index_of_episode + i + 1))
            action = np.random.choice(ALL_ACTIONS, p=action_probs)

            next_state, reward, terminated, truncated, info = env.step(action)

            # choose next action greedily
            next_action_probs = get_best_action_probs(q[next_state], epsilon=0)
            next_action = np.random.choice(ALL_ACTIONS, p=next_action_probs)

            q[state][action] += alpha * (reward + gamma * q[next_state][next_action] - q[state][action])

            state = next_state
            if terminated or truncated:
                break

    return q


def get_best_action_probs(state_actions, epsilon):
    action_probs = [epsilon / len(ALL_ACTIONS) for _ in ALL_ACTIONS]
    action_p = np.array([state_actions[a] for a in state_actions.keys()])
    if not np.all(action_p == action_p[0]): #if not all probabilities are the same
        best_a = max(state_actions, key=state_actions.get)
        action_probs[best_a] += 1 - epsilon
    else:
        action_probs = [1 / len(ALL_ACTIONS) for _ in ALL_ACTIONS]
    return action_probs


def create_greedy_policy(q):
    # pi: Dict(state -> action)
    pi = dict(zip(ALL_STATES,
                  [0 for _ in ALL_STATES]))
    for state in ALL_STATES:
        pi[state] = max(q[state], key=q[state].get)
    return pi


def q_learning_policy_iteration(episodes=1000, steps=10, alpha=0.1, gamma=0.95):
    pi = dict(zip(ALL_STATES,
                  [0 for _ in ALL_STATES]))
    q = None
    v_start_list = []

    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    counter = 0
    while True:
        counter += 1
        q = q_learning(env, q, episodes=episodes, alpha=alpha, gamma=gamma, index_of_episode=(counter - 1) * episodes)
        # pi_old = pi.copy()
        pi = create_greedy_policy(q)
        v_start = first_visit_monte_carlo_policy_evaluation(env, pi, episodes=1000)
        v_start_list.append(v_start)

        # pi_diff = {k: pi[k] for k in pi if k in pi_old and pi[k] != pi_old[k]}
        if counter >= steps:
            break

    return q, pi, v_start_list