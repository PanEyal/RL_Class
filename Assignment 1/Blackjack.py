import numpy as np
from tqdm import tqdm
import gymnasium as gym
import seaborn as sns

from itertools import product
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

# State representation: Tuple(sum_player, sum_dealer, usable_ace)
ALL_STATES = list(product(range(4, 33), range(1, 12), range(2)))
# Action representation: 0 -> stick, 1 -> hit
ALL_ACTIONS = list(range(2))


def transition_matrix(sample_size=10 ** 5, should_print=True):
    env = gym.make('Blackjack-v1', natural=False, sab=False)
    observation, info = env.reset()

    # n_s_a: Dict(state -> Dict(action -> Counter))
    n_s_a = dict(zip(ALL_STATES,
                     [dict(zip(ALL_ACTIONS,
                               [0. for _ in ALL_ACTIONS]))
                      for _ in ALL_STATES]))

    # n_s_a: Dict(state -> Dict(action -> Dict(states -> Counter)))
    n_s_a_ns = dict(zip(ALL_STATES,
                        [dict(zip(ALL_ACTIONS,
                                  [dict(zip(ALL_STATES,
                                            [0. for _ in ALL_STATES]))
                                   for _ in range(len(ALL_ACTIONS))]))
                         for _ in range(len(ALL_STATES))]))

    # r_s_a: Dict(state -> Dict(action -> Counter))
    r_s_a = dict(zip(ALL_STATES,
                     [dict(zip(ALL_ACTIONS,
                               [0. for _ in ALL_ACTIONS]))
                      for _ in ALL_STATES]))

    for _ in tqdm(range(sample_size)):
        action = env.action_space.sample()
        n_s_a[observation][action] += 1
        next_observation, reward, terminated, truncated, info = env.step(action)

        r_s_a[observation][action] += reward
        n_s_a_ns[observation][action][next_observation] += 1

        observation = next_observation

        if terminated or truncated:
            observation, info = env.reset()

    # tr_matrix: Dict(state -> Dict(action -> Dict(states -> Probability)))
    tr_matrix = dict(zip(ALL_STATES,
                         [dict(zip(ALL_ACTIONS,
                                   [dict(zip(ALL_STATES,
                                             [0. for _ in ALL_STATES]))
                                    for _ in range(len(ALL_ACTIONS))]))
                          for _ in range(len(ALL_STATES))]))

    for state in ALL_STATES:
        for action in ALL_ACTIONS:
            # Calculate average reward
            if n_s_a[state][action] != 0:
                r_s_a[state][action] = r_s_a[state][action] / n_s_a[state][action]
            for next_state in ALL_STATES:
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
        for state in ALL_STATES:
            action = pi[state]
            v[state] = r_s_a[state][action] \
                       + sum(tr_matrix[state][action][next_state] * v[next_state] for next_state in ALL_STATES)
    return v


def policy_improvement(v, pi, tr_matrix, r_s_a):
    for state in ALL_STATES:
        max_value = 0
        max_action = 0
        for action in ALL_ACTIONS:
            value = r_s_a[state][action] + sum(tr_matrix[state][action][next_state] * v[next_state]
                                               for next_state in ALL_STATES)
            if value > max_value:
                max_value = value
                max_action = action
        pi[state] = max_action
    return pi


def policy_iteration(tr_matrix, r_s_a, k=5):
    v = dict(zip(ALL_STATES,
                 [0 for _ in ALL_STATES]))
    pi = dict(zip(ALL_STATES,
                  [0 for _ in ALL_STATES]))
    while True:
        v = approximate_policy_evaluation(v, pi, tr_matrix, r_s_a, k)
        pi_old = pi
        pi = policy_improvement(v, pi, tr_matrix, r_s_a)
        if pi_old == pi:
            break
    return v, pi


def create_grids(v, pi, usable_ace=False):
    player_count, dealer_count = np.meshgrid(
        # players count, dealers face-up card
        np.arange(4, 22) if not usable_ace else np.arange(11, 22),
        np.arange(1, 12),
    )

    # create the value grid for plotting
    value = np.apply_along_axis(
        lambda obs: v[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    value_grid = player_count, dealer_count, value

    # create the policy grid for plotting
    policy_grid = np.apply_along_axis(
        lambda obs: pi[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    return value_grid, policy_grid


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


# ----------------------------- Main ----------------------------- #


# Get transition matrix and reward function
print("Generating transition matrix and reward function...")
tr_matrix, r_s_a = transition_matrix(sample_size=10 ** 6, should_print=False)

# Run policy iteration
print("Running policy iteration until convergence...")
v, pi = policy_iteration(tr_matrix, r_s_a, k=5)

for key, value in pi.items():
    print(f"{key} -> {'stick' if value == 0 else 'hit'}")

# state values & policy without usable ace (ace counts as 11)
value_grid, policy_grid = create_grids(v, pi, usable_ace=False)
fig = create_plots(value_grid, policy_grid, title="Without usable ace")
plt.show()
plt.clf()

# state values & policy with usable ace (ace counts as 11)
value_grid, policy_grid = create_grids(v, pi, usable_ace=True)
fig = create_plots(value_grid, policy_grid, title="With usable ace")
plt.show()
plt.clf()

