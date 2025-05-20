import numpy as np
import random

# Gridworld setup
grid_size = 3
states = [(i, j) for i in range(grid_size) for j in range(grid_size)]
actions = ['U', 'D', 'L', 'R']
discount = 0.99

action_effects = {
    'U': (-1, 0),
    'D': (1, 0),
    'L': (0, -1),
    'R': (0, 1)
}

perpendiculars = {
    'U': ['L', 'R'],
    'D': ['L', 'R'],
    'L': ['U', 'D'],
    'R': ['U', 'D']
}

def in_grid(x, y):
    return 0 <= x < grid_size and 0 <= y < grid_size

def get_next_states(state, action):
    i, j = state
    result = []

    # Intended direction
    di, dj = action_effects[action]
    ni, nj = i + di, j + dj
    if in_grid(ni, nj):
        result.append(((ni, nj), 0.8))
    else:
        result.append(((i, j), 0.8))

    # Perpendicular directions
    for perp in perpendiculars[action]:
        di, dj = action_effects[perp]
        ni, nj = i + di, j + dj
        if in_grid(ni, nj):
            result.append(((ni, nj), 0.1))
        else:
            result.append(((i, j), 0.1))

    return result

def policy_evaluation(policy, rewards, terminal_states, theta=1e-4):
    V = {s: 0 for s in states}
    while True:
        delta = 0
        new_V = V.copy()
        for state in states:
            if state in terminal_states:
                continue
            a = policy[state]
            value = 0
            for next_state, prob in get_next_states(state, a):
                value += prob * (rewards[state] + discount * V[next_state])
            new_V[state] = value
            delta = max(delta, abs(V[state] - new_V[state]))
        V = new_V
        if delta < theta:
            break
    return V

def policy_improvement(V, rewards, terminal_states):
    policy = {}
    for state in states:
        if state in terminal_states:
            continue
        best_action = None
        best_value = float('-inf')
        for action in actions:
            value = 0
            for next_state, prob in get_next_states(state, action):
                value += prob * (rewards[state] + discount * V[next_state])
            if value > best_value:
                best_value = value
                best_action = action
        policy[state] = best_action
    return policy

def policy_iteration(r):
    # Initial random policy
    policy = {s: random.choice(actions) for s in states}

    rewards = {
        (0, 0): r,
        (0, 1): -1,
        (0, 2): 10,
        (1, 0): -1,
        (1, 1): -1,
        (1, 2): -1,
        (2, 0): -1,
        (2, 1): -1,
        (2, 2): -1
    }

    terminal_states = [(0, 2)]

    is_policy_stable = False
    while not is_policy_stable:
        V = policy_evaluation(policy, rewards, terminal_states)
        new_policy = policy_improvement(V, rewards, terminal_states)
        is_policy_stable = (policy == new_policy)
        policy = new_policy

    return V, policy

def print_policy(policy):
    grid = [['' for _ in range(grid_size)] for _ in range(grid_size)]
    for (i, j) in states:
        grid[i][j] = policy.get((i, j), ' ')
    for row in grid:
        print(' '.join(row))

def print_values(V):
    for i in range(grid_size):
        row = [f"{V[(i, j)]:6.2f}" for j in range(grid_size)]
        print(' '.join(row))

# Run policy iteration for each value of r
if __name__ == '__main__':
    for r in [100, 3, 0, -3]:
        print(f"\n--- Policy Iteration Results for r = {r} ---")
        V, policy = policy_iteration(r)
        print("Value Function:")
        print_values(V)
        print("\nPolicy:")
        print_policy(policy)
