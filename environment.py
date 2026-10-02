"""
environment.py
-------------------------------------------------------------------
Reinforcement Learning 10x10 Grid Environment (Integrante 2 module)
Data with Roots Web Application.
-------------------------------------------------------------------
Grid specification (100 cells):
  1  'A' : agent start         (0, 0)
  1  'T' : target / goal       (9, 9)
  68 'o' : normal open path
  20 '#' : impassable wall
  10 'D' : danger zone (enterable, penalized)

Actions: 0 = Up, 1 = Down, 2 = Left, 3 = Right
States : flat index 0..99  (state = row * 10 + col)

Reward system
  Reach target T ............ +100
  Step on danger zone D ..... -20
  Step on normal path o ..... -1   (living cost, favours short routes)
  Hit a wall # .............. -5   (agent stays in place)
  Hit grid boundary ......... -5   (agent stays in place)

Episode ends when the target is reached or MAX_STEPS is exceeded.
"""

import numpy as np

GRID_MAP = [
    ['A', 'o', 'o', '#', 'o', 'o', 'o', 'o', 'o', 'o'],
    ['o', '#', 'o', '#', 'o', 'D', 'o', '#', '#', 'o'],
    ['o', '#', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o'],
    ['o', 'o', 'D', '#', '#', '#', 'o', 'D', 'o', 'o'],
    ['#', '#', 'o', 'o', 'o', 'o', 'o', 'o', '#', 'o'],
    ['o', 'o', 'o', 'D', '#', 'o', 'o', 'o', 'o', 'o'],
    ['o', '#', '#', 'o', 'o', 'D', 'D', '#', 'o', 'o'],
    ['o', 'D', 'o', 'o', '#', 'o', 'o', 'D', 'o', 'o'],
    ['o', 'o', 'o', '#', 'o', 'o', 'D', 'o', 'o', 'o'],
    ['o', '#', 'o', 'o', 'D', 'o', 'o', 'o', '#', 'T'],
]

REWARDS = {
    'T': 100.0,        # reaching the goal
    'D': -20.0,        # danger zone
    '#': -5.0,         # wall collision
    'o': -1.0,         # normal step
    'boundary': -5.0,  # grid boundary collision
}

ACTION_NAMES = ['Up', 'Down', 'Left', 'Right']
ACTION_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


class MazeEnvironment10x10:
    """10x10 grid world queried by the Q-Learning agent."""

    def __init__(self):
        self.rows = 10
        self.cols = 10
        self.actions = ACTION_NAMES          # names, index = action id
        self.action_space = [0, 1, 2, 3]
        self.rewards = dict(REWARDS)
        self.grid = np.array(GRID_MAP)
        self.start_pos = (0, 0)
        self.target_pos = (9, 9)
        self.max_steps = 150
        self.agent_pos = self.start_pos
        self.current_step = 0

    # ---- state helpers -------------------------------------------------
    def get_state_index(self, pos):
        """(row, col) -> flat index 0..99."""
        return pos[0] * self.cols + pos[1]

    def get_pos_from_index(self, state_idx):
        """Flat index -> (row, col)."""
        return (state_idx // self.cols, state_idx % self.cols)

    def get_cell_counts(self):
        """Quantity of each cell type (must be 1 A, 1 T, 68 o, 20 #, 10 D)."""
        flat = self.grid.flatten().tolist()
        return {
            'A': flat.count('A'),
            'T': flat.count('T'),
            'o': flat.count('o'),
            '#': flat.count('#'),
            'D': flat.count('D'),
            'Total': len(flat),
        }

    # ---- dynamics ------------------------------------------------------
    def reset(self):
        """Puts the agent on 'A' and returns the initial state index."""
        self.agent_pos = self.start_pos
        self.current_step = 0
        return self.get_state_index(self.agent_pos)

    def step(self, action_idx):
        """
        Executes one action.
        Returns (next_state, reward, done, cell_type) where cell_type is one of
        'o', 'D', 'T', '#', 'boundary' (the cell the agent tried to enter).
        """
        self.current_step += 1
        r, c = self.agent_pos
        dr, dc = ACTION_DELTAS[int(action_idx)]
        nr, nc = r + dr, c + dc
        done = False

        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            cell_type = 'boundary'
            reward = self.rewards['boundary']
        else:
            cell_type = str(self.grid[nr, nc])
            if cell_type == '#':
                reward = self.rewards['#']
            else:
                self.agent_pos = (nr, nc)
                reward = self.rewards.get(cell_type, self.rewards['o'])
                done = cell_type == 'T'

        if self.current_step >= self.max_steps:
            done = True

        return self.get_state_index(self.agent_pos), reward, done, cell_type


if __name__ == "__main__":
    env = MazeEnvironment10x10()
    print("--- 10x10 CELL COUNT VERIFICATION ---")
    for key, val in env.get_cell_counts().items():
        print(f"{key}: {val}")
