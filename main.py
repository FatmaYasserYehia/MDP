import numpy as np

# Define grid size
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

def value_iteration(r, threshold=1e-4):
    V = {s: 0 for s in states}
    policy = {s: 'U' for s in states}

    # Define state-dependent rewards
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

    while True:
        delta = 0
        new_V = V.copy()
        for state in states:
            if state in terminal_states:
                continue  # Skip updating terminal states

            max_val = float('-inf')
            best_action = None
            for action in actions:
                total = 0
                for (next_state, prob) in get_next_states(state, action):
                    total += prob * (rewards[state] + discount * V[next_state])
                if total > max_val:
                    max_val = total
                    best_action = action
            new_V[state] = max_val
            policy[state] = best_action
            delta = max(delta, abs(V[state] - new_V[state]))
        V = new_V
        if delta < threshold:
            break

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

# Example run
if __name__ == '__main__':
    for r in [100, 3, 0, -3]:
        print(f"\n--- Results for r = {r} ---")
        V, policy = value_iteration(r)
        print("Value Function:")
        print_values(V)
        print("\nPolicy:")
        print_policy(policy)