import numpy as np
import gymnasium as gym

from itertools import product

# Here we will implement the policy iteration algorithm for the frozen_lake environment.

# State representation: Integer from 0 to 15
ALL_STATES = list(range(16))
# Action representation: 0 -> left, 1 -> down, 2 -> right, 3 -> up
ALL_ACTIONS = list(range(4))

ITERATION_LIMIT = 500


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


def q_learning(env, q, episodes=1000, alpha=0.1, gamma=1.0, epsilon=0.3):
    # q: Dict(state -> Dict(action -> value))
    if q is None:
        q = dict(zip(ALL_STATES,
                     [dict(zip(ALL_ACTIONS,
                               [0. for _ in ALL_ACTIONS]))
                      for _ in ALL_STATES]))

    for _ in range(episodes):
        state, info = env.reset()
        action_probs = get_best_action_probs(q[state], epsilon)
        action = np.random.choice(ALL_ACTIONS, p=action_probs)
        while True:
            next_state, reward, terminated, truncated, info = env.step(action)

            action_probs = get_best_action_probs(q[next_state], epsilon)
            next_action = np.random.choice(ALL_ACTIONS, p=action_probs)
            q[state][action] += alpha * (reward + gamma * q[next_state][next_action] - q[state][action])
            state = next_state
            action = next_action

            if terminated or truncated:
                break

    return q


def get_best_action_probs(state_actions, epsilon):
    action_probs = [epsilon / len(ALL_ACTIONS) for _ in ALL_ACTIONS]
    best_a = max(state_actions, key=state_actions.get)
    action_probs[best_a] += 1 - epsilon
    return action_probs


def create_greedy_policy(q):
    # pi: Dict(state -> action)
    pi = dict(zip(ALL_STATES,
                  [0 for _ in ALL_STATES]))
    for state in ALL_STATES:
        pi[state] = max(q[state], key=q[state].get)
    return pi


def q_learning_policy_iteration(episodes=1000, alpha=0.1, gamma=0.95, epsilon=0.3):
    pi = dict(zip(ALL_STATES,
                  [0 for _ in ALL_STATES]))
    q = None
    v_start_list = []

    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    counter = 0
    while True:
        counter += 1
        q = q_learning(env, q, episodes=episodes, alpha=alpha, gamma=gamma, epsilon=epsilon)
        pi_old = pi.copy()
        pi = create_greedy_policy(q)
        v_start = first_visit_monte_carlo_policy_evaluation(env, pi, episodes=1000, gamma=gamma)
        v_start_list.append(v_start)

        pi_diff = {k: pi[k] for k in pi if k in pi_old and pi[k] != pi_old[k]}
        if len(pi_diff) == 0:
            print(f"Policy iteration converged after {counter} iterations.")
            break
        if counter >= ITERATION_LIMIT:
            print(f"Policy iteration did not converge after {counter} iterations.")
            break
        print(f"counter: {counter}, diff in pi: {len(pi_diff)}", end='\r')

    return q, pi, v_start_list