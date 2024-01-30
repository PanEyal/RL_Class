import numpy as np
from tqdm import tqdm
import gymnasium as gym
import seaborn as sns

from itertools import product
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

# State representation: Tuple(sum_player, sum_dealer, usable_ace)
ALL_STATES = list(product(range(4, 22), range(1, 12), range(2)))
# Action representation: 0 -> stick, 1 -> hit
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

    # r_s_a: Dict(state -> Dict(action -> Counter))
    r_s_a = dict(zip(ALL_STATES_WITH_END,
                     [dict(zip(ALL_ACTIONS, [0. for _ in ALL_ACTIONS])) for _ in ALL_STATES]
                     + [dict(zip(ALL_ACTIONS, [1. for _ in ALL_ACTIONS]))]
                     + [dict(zip(ALL_ACTIONS, [-1. for _ in ALL_ACTIONS]))]
                     + [dict(zip(ALL_ACTIONS, [0. for _ in ALL_ACTIONS]))]))

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
    return tr_matrix, r_s_a


def approximate_policy_evaluation(v, pi, tr_matrix, r_s_a, k=5):
    for _ in range(k):
        for state in ALL_STATES_WITH_END:
            action = pi[state]
            v[state] = r_s_a[state][action] \
                       + sum(tr_matrix[state][action][next_state] * v[next_state] for next_state in ALL_STATES_WITH_END)
    return v


def policy_improvement(v, pi, tr_matrix, r_s_a):
    for state in ALL_STATES_WITH_END:
        max_value = 0
        max_action = 0
        for action in ALL_ACTIONS:
            value = r_s_a[state][action] + sum(tr_matrix[state][action][next_state] * v[next_state]
                                               for next_state in ALL_STATES_WITH_END)
            if value > max_value:
                max_value = value
                max_action = action
        pi[state] = max_action
    return pi


def policy_iteration(tr_matrix, r_s_a, k=5, v=None, pi=None, states_to_print=None):
    if v is None:
        v = dict(zip(ALL_STATES_WITH_END,
                     [0 for _ in ALL_STATES_WITH_END]))
    if pi is None:
        pi = dict(zip(ALL_STATES_WITH_END,
                      [0 for _ in ALL_STATES_WITH_END]))

    values_avg = [0.]
    counter = 0
    while True:
        counter += 1
        v = approximate_policy_evaluation(v, pi, tr_matrix, r_s_a, k)
        values_avg.append(np.mean(list(v.values())))
        pi_old = pi.copy()
        pi = policy_improvement(v, pi, tr_matrix, r_s_a)

        if states_to_print is not None:
            for state, chosen_actions in states_to_print.items():
                chosen_actions.append(pi[state])

        pi_diff = {k: pi[k] for k in pi if k in pi_old and pi[k] != pi_old[k]}
        if len(pi_diff) == 0:
            print(f"Policy iteration converged after {counter} iterations.")
            break
        else:
            print(f"counter: {counter}, diff in pi: {len(pi_diff)}", end='\r')

    return v, pi, values_avg


# def create_grids(v, pi, usable_ace=False):
#     player_count, dealer_count = np.meshgrid(
#         # players count, dealers face-up card
#         np.arange(4, 21) if not usable_ace else np.arange(11, 22),
    #     np.arange(1, 12),
    # )
    #
    # # create the value grid for plotting
    # value = np.apply_along_axis(
    #     lambda obs: v[(obs[0], obs[1], usable_ace)],
    #     axis=2,
    #     arr=np.dstack([player_count, dealer_count]),
    # )
    # value_grid = player_count, dealer_count, value
    #
    # # create the policy grid for plotting
    # policy_grid = np.apply_along_axis(
    #     lambda obs: pi[(obs[0], obs[1], usable_ace)],
    #     axis=2,
    #     arr=np.dstack([player_count, dealer_count]),
    # )
    # return value_grid, policy_grid


def create_plots(value_grid, policy_grid, title: str):
    """Creates a plot using a value and policy grid."""
    # create a new figure with 2 subplots (left: state values, right: policy)
    player_count, dealer_count, value = value_grid
    fig = plt.figure(figsize=(15, 7))
    fig.suptitle(title, fontsize=16)

    # plot the state values
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(
        player_count,
        dealer_count,
        value,
        rstride=1,
        cstride=1,
        cmap="viridis",
        edgecolor="none",
    )
    plt.xticks(value_grid[0][1], value_grid[0][1])
    plt.yticks(range(1, 12), ["A"] + list(range(2, 12)))
    ax1.set_title(f"State values: {title}")
    ax1.set_xlabel("Player sum")
    ax1.set_ylabel("Dealer showing")
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel("Value", fontsize=14, rotation=90)
    ax1.view_init(20, 220)

    # plot the policy
    fig.add_subplot(1, 2, 2)
    ax2 = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
    ax2.set_title(f"Policy: {title}")
    ax2.set_xlabel("Player sum")
    ax2.set_ylabel("Dealer showing")
    ax2.set_xticklabels(value_grid[0][1])
    ax2.set_yticklabels(["A"] + list(range(2, 12)), fontsize=12)

    # add a legend
    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
        Patch(facecolor="grey", edgecolor="black", label="Stick"),
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
    return fig


# # ----------------------------- Main ----------------------------- #
#
#
# # Get transition matrix and reward function
# print("Generating transition matrix and reward function...")
# tr_matrix, r_s_a = transition_matrix(sample_size=10 ** 5, should_print=False)
#
# pi = dict()
# for state in ALL_STATES:
#     pi[state] = 1 if state[0] < 21 else 0
# for state in [WIN, LOSE, DRAW]:
#     pi[state] = 0
#
# SIZE = 5
# indices = np.random.choice(np.arange(len(ALL_STATES)), size=SIZE, replace=False)
# states_indices = [ALL_STATES[i] for i in indices]
# states_to_print = dict(zip(states_indices, [[] for _ in range(SIZE)]))
#
# # Run policy iteration
# print("Running policy iteration until convergence...")
# v, pi, value_avg = policy_iteration(tr_matrix, r_s_a, k=5, pi=pi, states_to_print=states_to_print)
#
# for state, chosen_actions in states_to_print.items():
#     plt.plot(np.arange(1, len(states_to_print[state]) + 1), chosen_actions, 'o-',
#              label=f"state ({state[0]}, {state[1]}, {state[2]})")
# plt.xlabel("iteration")
# plt.ylabel("action")
# plt.yticks([0, 1], ["stick", "hit"])
# plt.legend()
# plt.show()
# plt.clf()


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
#
#
# plt.plot(np.arange(len(value_avg)), value_avg, 'o-')
# plt.title("Average value function over iterations")
# plt.xlabel("iteration")
# plt.ylabel("value")
# plt.show()
# plt.clf()
#
# player_count, dealer_count = np.meshgrid(
#     # players count, dealers face-up card
#     np.arange(4, 21),
#     np.arange(1, 12),
# )
# # create the policy grid for plotting
# pi_grid = np.apply_along_axis(
#     lambda obs: pi[(obs[0], obs[1], 0)],
#     axis=2,
#     arr=np.dstack([player_count, dealer_count]),
# )
# # plot the policy for unusable ace
# plt.figure()
# sns.heatmap(pi_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False, xticklabels=range(4, 21), yticklabels=["A"] + list(range(2, 12)))
# plt.title("Starting policies for unusable ace", fontsize=20)
# plt.xlabel("Player sum")
# plt.ylabel("Dealer showing")
#
# # add a legend
# legend_elements = [
#     Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
#     Patch(facecolor="grey", edgecolor="black", label="Stick"),
# ]
# plt.legend(handles=legend_elements)
# plt.show()
# plt.clf()
