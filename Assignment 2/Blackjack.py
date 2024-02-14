import numpy as np
from tqdm import tqdm
import gymnasium as gym

from itertools import product

# Here we will implement the policy iteration algorithm for the Blackjack environment.

# State representation: Tuple(sum_player, sum_dealer, usable_ace)
ALL_STATES = list(product(range(4, 22), range(1, 12), range(2)))
# Action representation: 0 -> stick, 1 -> hit, 2 -> special action only from terminating states
ALL_ACTIONS = list(range(2))

# enum for finished states:
WIN = "win"
LOSE = "lose"
DRAW = "draw"
ALL_STATES_WITH_END = ALL_STATES + [WIN, LOSE, DRAW]


def transition_matrix(sample_size=10 ** 5, should_print=True):
    env = gym.make('Blackjack-v1', natural=False, sab=False)
    observation, info = env.reset()

    # n_s_a: Dict(state -> Dict(action -> Counter))
    n_s_a = dict(zip(ALL_STATES_WITH_END,
                     [dict(zip(ALL_ACTIONS,
                               [0. for _ in ALL_ACTIONS]))
                      for _ in ALL_STATES_WITH_END]))

    # n_s_a: Dict(state -> Dict(action -> Dict(states -> Counter)))
    n_s_a_ns = dict(zip(ALL_STATES_WITH_END,
                        [dict(zip(ALL_ACTIONS,
                                  [dict(zip(ALL_STATES_WITH_END,
                                            [0. for _ in ALL_STATES_WITH_END]))
                                   for _ in range(len(ALL_ACTIONS))]))
                         for _ in range(len(ALL_STATES_WITH_END))]))

    for _ in tqdm(range(sample_size)):
        action = env.action_space.sample()
        n_s_a[observation][action] += 1
        next_observation, reward, terminated, truncated, info = env.step(action)

        if terminated:
            if reward > 0:
                n_s_a_ns[observation][action][WIN] += 1
            elif reward == 0:
                n_s_a_ns[observation][action][DRAW] += 1
            else:  # reward == -1
                n_s_a_ns[observation][action][LOSE] += 1

        else:
            n_s_a_ns[observation][action][next_observation] += 1
            observation = next_observation

        if terminated or truncated:
            observation, info = env.reset()

    # tr_matrix: Dict(state -> Dict(action -> Dict(states -> Probability)))
    tr_matrix = dict(zip(ALL_STATES_WITH_END,
                         [dict(zip(ALL_ACTIONS,
                                   [dict(zip(ALL_STATES_WITH_END,
                                             [0. for _ in ALL_STATES_WITH_END]))
                                    for _ in range(len(ALL_ACTIONS))]))
                          for _ in range(len(ALL_STATES_WITH_END))]))

    for state in ALL_STATES_WITH_END:
        for action in ALL_ACTIONS:
            for next_state in ALL_STATES_WITH_END:
                # Calculate transition probability
                if n_s_a[state][action] != 0:
                    p = n_s_a_ns[state][action][next_state] / n_s_a[state][action]
                    tr_matrix[state][action][next_state] = p
                    if p != 0 and should_print:
                        print(f"s: {state}, a: {action}, s': {next_state}, P: {np.round(p, 3)}")

    env.close()
    return tr_matrix


def reward_function():
    # r_s_a: Dict(state -> Dict(action -> Counter))
    r_s_a = dict(zip(ALL_STATES_WITH_END,
                     [dict(zip(ALL_ACTIONS, [0. for _ in ALL_ACTIONS])) for _ in ALL_STATES]
                     + [dict(zip(ALL_ACTIONS, [1. for _ in ALL_ACTIONS]))]
                     + [dict(zip(ALL_ACTIONS, [-1. for _ in ALL_ACTIONS]))]
                     + [dict(zip(ALL_ACTIONS, [0. for _ in ALL_ACTIONS]))]))
    return r_s_a


def play_episode(env, pi):
    observation, info = env.reset()
    episode = []
    while True:
        action = pi[observation]
        next_observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            if reward > 0:
                observation = WIN
            elif reward == 0:
                observation = DRAW
            else:  # reward == -1
                observation = LOSE
            episode.append((observation, reward))
            break
        else:
            episode.append((observation, reward))
            observation = next_observation

    return episode


def policy_improvement(v, pi, tr_matrix, r_s_a):
    for state in ALL_STATES_WITH_END:
        max_value = None
        max_action = None
        for action in ALL_ACTIONS:
            value = r_s_a[state][action] + sum(tr_matrix[state][action][next_state] * v[next_state]
                                               for next_state in ALL_STATES_WITH_END)
            if max_action is None or value > max_value:
                max_value = value
                max_action = action
        pi[state] = max_action
    return pi


def monte_carlo_policy_evaluation(env, pi, episodes=1000, first_visit=True, gamma=1.0):
    n = dict(zip(ALL_STATES_WITH_END,
                 [0. for _ in ALL_STATES_WITH_END]))
    s = dict(zip(ALL_STATES_WITH_END,
                 [0. for _ in ALL_STATES_WITH_END]))
    v = dict(zip(ALL_STATES_WITH_END,
                 [0. for _ in ALL_STATES_WITH_END]))

    for _ in range(episodes):
        episode = play_episode(env, pi)
        g = 0
        for i, (state, reward) in enumerate(reversed(episode), 1):
            g = gamma * g + reward

            if first_visit and state in [other_state for other_state, _ in episode[:-i]]:
                continue
            n[state] += 1
            s[state] += g
            v[state] = s[state] / n[state]

    return v


def calc_v_avg(v):
    sum = 0
    counter = 0
    for state in list(product(range(13, 17), range(7, 9), range(0, 1))):
        sum += v[state]
        counter += 1
    return sum / counter


def monte_carlo_policy_iteration(tr_matrix, r_s_a, pi=None, episodes=1000, first_visit=True, gamma=1.0):
    if pi is None:
        pi = dict(zip(ALL_STATES_WITH_END,
                      [0 for _ in ALL_STATES_WITH_END]))

    env = gym.make('Blackjack-v1', natural=False, sab=False)
    values_avg = []
    counter = 0
    while True:
        counter += 1
        v = monte_carlo_policy_evaluation(env, pi, episodes=episodes, first_visit=first_visit, gamma=gamma)
        values_avg.append(calc_v_avg(v))
        pi_old = pi.copy()
        pi = policy_improvement(v, pi, tr_matrix, r_s_a)

        pi_diff = {k: pi[k] for k in pi if k in pi_old and pi[k] != pi_old[k]}
        if len(pi_diff) == 0:
            print(f"Policy iteration converged after {counter} iterations.")
            break
        if counter >= 100:
            print(f"Policy iteration did not converge after {counter} iterations.")
            break
        print(f"counter: {counter}, diff in pi: {len(pi_diff)}", end='\r')

    return v, pi, values_avg


def sarsa(env, episodes=1000, alpha=0.1, gamma=1.0, epsilon=0.3):
    # q: Dict(state -> Dict(action -> Value))
    q = dict(zip(ALL_STATES_WITH_END,
                 [dict(zip(ALL_ACTIONS,
                           [0. for _ in ALL_ACTIONS]))
                  for _ in ALL_STATES_WITH_END]))

    for _ in range(episodes):
        state, info = env.reset()
        policy_s = get_probs(q[state], epsilon)
        action = np.random.choice(ALL_ACTIONS, p=policy_s)
        while True:
            next_state, reward, terminated, truncated, info = env.step(action)
            if not terminated and not truncated:
                policy_s = get_probs(q[next_state], epsilon)
                next_action = np.random.choice(ALL_ACTIONS, p=policy_s)
                q[state][action] = q[state][action] + alpha * (
                        reward + gamma * q[next_state][next_action] - q[state][action])
                state = next_state
                action = next_action
            else:
                q[state][action] = q[state][action] + alpha * (reward - q[state][action])
                break

    return q


def get_probs(q_s, epsilon):
    policy_s = np.ones(len(ALL_ACTIONS)) * epsilon / len(ALL_ACTIONS)
    best_a = max(q_s, key=q_s.get)
    policy_s[best_a] = 1 - epsilon + (epsilon / len(ALL_ACTIONS))
    return policy_s


def create_greedy_policy(q):
    pi = dict(zip(ALL_STATES_WITH_END,
                  [0 for _ in ALL_STATES_WITH_END]))
    for state in ALL_STATES_WITH_END:
        pi[state] = np.argmax(q[state])
    return pi


def calc_q_avg(q):
    sum = 0
    counter = 0
    for state in list(product(range(13, 17), range(7, 9), range(0, 1))):
        for action in ALL_ACTIONS:
            sum += q[state][action]
            counter += 1
    return sum / counter


def sarsa_policy_iteration(episodes=1000, alpha=0.1, gamma=1.0, epsilon=0.3):
    pi = dict(zip(ALL_STATES_WITH_END,
                  [0 for _ in ALL_STATES_WITH_END]))

    env = gym.make('Blackjack-v1', natural=False, sab=False)
    values_avg = []
    counter = 0
    while True:
        counter += 1
        q = sarsa(env, episodes=episodes, alpha=alpha, gamma=gamma, epsilon=epsilon)
        values_avg.append(calc_q_avg(q))
        pi_old = pi.copy()
        pi = create_greedy_policy(q)

        pi_diff = {k: pi[k] for k in pi if k in pi_old and pi[k] != pi_old[k]}
        if len(pi_diff) == 0:
            print(f"Policy iteration converged after {counter} iterations.")
            break
        if counter >= 100:
            print(f"Policy iteration did not converge after {counter} iterations.")
            break
        print(f"counter: {counter}, diff in pi: {len(pi_diff)}", end='\r')

    return q, pi, values_avg


def play(pi):
    print("Playing with chosen policy...")
    env = gym.make('Blackjack-v1', natural=False, sab=False)
    observation, info = env.reset()

    while True:
        action = pi[observation]
        next_observation, reward, terminated, truncated, info = env.step(action)
        print(f"s: {observation}, a: {action}, s': {next_observation}, r: {reward}")
        observation = next_observation
        if terminated or truncated:
            break

    env.close()
    if reward > 0:
        print("Player won!")
    elif reward == 0:
        print("Draw!")
    else:
        print("Player lost!")


def value_function_q3(v, s):
    print(f'For state: ({s[0]}, {s[1]}, {"hit" if s[2] else "stick"}) -> value: {v[s]}')
    return v[s]
