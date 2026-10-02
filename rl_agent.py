
"""
Q-Learning Agent Implementation for Grid World Navigation.
Handles epsilon-greedy exploration/exploitation, incremental Bellman updates,
policy evaluation, step-by-step path recording, and Q-value formatting.
"""

import numpy as np
import random
from rl_environment import GridWorld10x10

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.env = env
        self.n_states = env.rows * env.cols  # 100 states
        self.n_actions = len(env.actions)    # 4 actions
        self.q_table = np.zeros((self.n_states, self.n_actions))
        
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

    def select_action(self, state, explore=True):
        """Selects an action using Epsilon-Greedy strategy or greedy exploitation."""
        if explore and random.uniform(0, 1) < self.epsilon:
            return random.choice(self.env.action_space)
        else:
            # Break ties randomly when maximum Q-values are identical
            q_values = self.q_table[state]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return int(random.choice(best_actions))

    def train(self, episodes=300):
        """
        Executes episodic Q-Learning training loop.
        Applies Bellman update equation:
          Q(s, a) = Q(s, a) + alpha * [reward + gamma * max_a' Q(s', a') - Q(s, a)]
        """
        successful_episodes = 0
        total_rewards = []

        for ep in range(episodes):
            state = self.env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                action = self.select_action(state, explore=True)
                next_state, reward, done, cell_type = self.env.step(action)

                best_next = np.max(self.q_table[next_state])
                td_target = reward + (self.gamma * best_next if not done or cell_type == 'T' else 0.0)
                td_error = td_target - self.q_table[state, action]
                self.q_table[state, action] += self.alpha * td_error

                state = next_state
                ep_reward += reward

                if cell_type == 'T':
                    successful_episodes += 1

            total_rewards.append(ep_reward)

            # Decay epsilon exploration rate
            if self.epsilon > self.epsilon_min:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        metrics = {
            'total_episodes': episodes,
            'successful_episodes': successful_episodes,
            'success_rate': round((successful_episodes / episodes) * 100, 2),
            'avg_reward': round(float(np.mean(total_rewards)), 2),
            'final_epsilon': round(float(self.epsilon), 4)
        }
        return metrics

    def evaluate(self):
        """
        Evaluates the learned optimal policy without exploration (pure exploitation).
        Records the step-by-step history and trajectories.
        """
        state = self.env.reset()
        done = False
        step_history = []
        path = [state]
        total_eval_reward = 0.0
        step_count = 0
        target_reached = False

        while not done and step_count < self.env.max_steps:
            step_count += 1
            action = self.select_action(state, explore=False)
            action_name = self.env.actions[action]
            next_state, reward, done, cell_type = self.env.step(action)
            total_eval_reward += reward

            step_history.append({
                'step': step_count,
                'state': state,
                'action': action_name,
                'next_state': next_state,
                'cell_type': cell_type,
                'reward': reward
            })

            state = next_state
            path.append(state)

            if cell_type == 'T':
                target_reached = True
                break

        evaluation_results = {
            'target_reached': target_reached,
            'total_moves': step_count,
            'total_reward': round(total_eval_reward, 2),
            'step_history': step_history,
            'path': path
        }
        return evaluation_results

    def get_q_table_records(self):
        """Prepares Q-table matrix for frontend rendering."""
        records = []
        for s in range(self.n_states):
            r, c = self.env.get_pos_from_index(s)
            records.append({
                'state': s,
                'row': r,
                'col': c,
                'cell_type': self.env.grid[r, c],
                'up': round(float(self.q_table[s, 0]), 3),
                'down': round(float(self.q_table[s, 1]), 3),
                'left': round(float(self.q_table[s, 2]), 3),
                'right': round(float(self.q_table[s, 3]), 3),
                'best_action': self.env.actions[int(np.argmax(self.q_table[s]))]
            })
        return records