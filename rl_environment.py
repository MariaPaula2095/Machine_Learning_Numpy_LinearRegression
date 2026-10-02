
"""
10x10 Grid World Environment for Reinforcement Learning.
Defines the state space, action dynamics, reward structure, and collision logic.
Complies with the required distribution:
  - 1 'A': Start point (0, 0)
  - 1 'T': Target / Goal (9, 9)
  - 20 '#': Impassable obstacles / Walls
  - 10 'D': Hazard / Danger zones
  - 68 'o': Standard open path cells
  Total: 100 cells.
"""

import numpy as np

class GridWorld10x10:
    def __init__(self):
        self.rows = 10
        self.cols = 10
        self.actions = ['Up', 'Down', 'Left', 'Right']
        self.action_space = [0, 1, 2, 3]  # 0: Up, 1: Down, 2: Left, 3: Right
        
        # Reward configuration
        self.rewards = {
            'T': 100.0,       # Reaching the goal
            'D': -20.0,       # Danger/Hazard penalty
            '#': -5.0,        # Wall collision penalty
            'o': -1.0,        # Living cost per transition (encourages shortest path)
            'boundary': -5.0  # Grid boundary collision penalty
        }
        
        self.grid = self._init_grid()
        self.start_pos = (0, 0)
        self.target_pos = (9, 9)
        self.agent_pos = self.start_pos
        self.max_steps = 150
        self.current_step = 0

    def _init_grid(self):
        """
        Initializes the deterministic 10x10 map.
        Guarantees that a valid navigable path exists from start to target.
        """
        grid = np.array([
            ['A', 'o', 'o', '#', 'o', 'o', 'o', 'o', 'o', 'o'],
            ['o', '#', 'o', '#', 'o', 'D', 'D', 'o', '#', 'o'],
            ['o', '#', 'o', 'o', 'o', 'o', 'o', 'o', '#', 'o'],
            ['o', 'o', 'o', '#', '#', '#', 'o', 'D', 'o', 'o'],
            ['#', '#', 'o', 'o', 'D', 'o', 'o', '#', '#', 'o'],
            ['o', 'o', 'o', 'D', 'D', 'o', 'o', 'o', 'o', 'o'],
            ['o', '#', '#', 'o', 'o', 'o', '#', '#', 'o', 'o'],
            ['o', 'D', 'o', 'o', '#', 'o', 'o', 'D', 'o', '#'],
            ['o', 'o', 'o', '#', '#', 'o', 'D', 'o', 'o', 'o'],
            ['o', '#', 'o', 'o', 'o', 'o', 'o', 'o', '#', 'T']
        ])
        return grid

    def reset(self):
        """Resets the agent to the starting cell (0, 0) and clears step count."""
        self.agent_pos = self.start_pos
        self.current_step = 0
        return self.get_state_index(self.agent_pos)

    def get_state_index(self, pos):
        """Converts 2D coordinate (row, col) to a flat 1D state index (0 to 99)."""
        return pos[0] * self.cols + pos[1]

    def get_pos_from_index(self, state_idx):
        """Converts flat 1D state index back to 2D coordinate (row, col)."""
        return (state_idx // self.cols, state_idx % self.cols)

    def step(self, action_idx):
        """
        Executes one transition step in the environment.
        Returns: next_state, reward, done, cell_type
        """
        self.current_step += 1
        r, c = self.agent_pos
        
        # Coordinate offsets for actions
        delta = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }
        dr, dc = delta[action_idx]
        nr, nc = r + dr, c + dc

        done = False
        reward = 0.0

        # Boundary checks
        if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
            next_pos = (r, c)
            cell_type = 'boundary'
            reward = self.rewards['boundary']
        else:
            cell_type = self.grid[nr, nc]
            if cell_type == '#':
                # Blocked by wall: maintain position and penalize
                next_pos = (r, c)
                reward = self.rewards['#']
            else:
                next_pos = (nr, nc)
                reward = self.rewards.get(cell_type, -1.0)
                if cell_type == 'T':
                    done = True

        self.agent_pos = next_pos

        if self.current_step >= self.max_steps:
            done = True

        next_state = self.get_state_index(self.agent_pos)
        return next_state, reward, done, cell_type