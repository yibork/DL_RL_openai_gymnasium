import numpy as np
import gymnasium
import random
import matplotlib.pyplot as plt
import json
import os
import itertools
import pandas as pd

NUM_BINS = (6, 6, 12, 12)
STATE_BOUNDS = list(zip(
    [-4.8, -5, -0.418, -5],
    [4.8, 5, 0.418, 5]
))

def discretize_state(state, bins=NUM_BINS, bounds=STATE_BOUNDS):
    discretized = []
    for i in range(len(state)):
        if state[i] <= bounds[i][0]:
            bin_index = 0
        elif state[i] >= bounds[i][1]:
            bin_index = bins[i] - 1
        else:
            scale = (state[i] - bounds[i][0]) / (bounds[i][1] - bounds[i][0])
            bin_index = int(scale * bins[i])
            if bin_index >= bins[i]:
                bin_index = bins[i] - 1
        discretized.append(bin_index)
    return tuple(discretized)

class QLearningAgent:
    def __init__(self, env, bins, alpha=0.1, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.env = env
        self.bins = bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.Q = np.zeros(bins + (env.action_space.n,))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state])

    def learn(self, current_state, action, reward, next_state, done):
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state][best_next_action] * (not done)
        td_delta = td_target - self.Q[current_state][action]
        self.Q[current_state][action] += self.alpha * td_delta

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_as_json(self, filename='q_table.json'):
        q_table_list = self.Q.tolist()
        with open(filename, 'w') as f:
            json.dump(q_table_list, f)
        print(f"Q-table saved to {filename}.")

    def load_from_json(self, filename='q_table.json'):
        if not os.path.exists(filename):
            print(f"File {filename} not found. Starting with a fresh Q-table.")
            return
        with open(filename, 'r') as f:
            q_table_list = json.load(f)
        self.Q = np.array(q_table_list)
        print(f"Q-table loaded from {filename}.")


def train_agent(env, agent, episodes=10000, max_steps=200, window_size=100):
    rewards = []
    moving_avg_rewards = []
    first_five_q_tables = []
    last_five_q_tables = []

    for episode in range(episodes):
        state, info = env.reset()
        state = discretize_state(state)
        total_reward = 0

        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = discretize_state(next_state)

            if done and step < max_steps - 1:
                reward = -100

            agent.learn(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                break

        rewards.append(total_reward)

        if episode < 5:
            first_five_q_tables.append(agent.Q.copy())

        if len(last_five_q_tables) < 5:
            last_five_q_tables.append(agent.Q.copy())
        else:
            last_five_q_tables.pop(0)
            last_five_q_tables.append(agent.Q.copy())

        if episode >= window_size:
            moving_avg = np.mean(rewards[-window_size:])
            moving_avg_rewards.append(moving_avg)
        else:
            moving_avg_rewards.append(np.mean(rewards))

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            print(f"Episode {episode + 1}: Average Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}")

        if episode + 1 == 1000:
            print(f"Solved after {episode + 1} episodes!")
            agent.save_as_json('q_table_final.json')
            break

    return rewards, moving_avg_rewards, first_five_q_tables, last_five_q_tables

def evaluate_agent(env, agent, episodes=100, max_steps=200):
    total_rewards = []
    agent.epsilon = 0.0

    for episode in range(episodes):
        state, info = env.reset()
        state = discretize_state(state)
        total_reward = 0

        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = discretize_state(next_state)

            state = next_state
            total_reward += reward

            if done:
                break

        total_rewards.append(total_reward)

    avg_reward = np.mean(total_rewards)
    print(f"Average Reward over {episodes} episodes: {avg_reward:.2f}")
    return total_rewards

def visualize_agent(env, agent, episodes=5, max_steps=200):
    for episode in range(episodes):
        state, info = env.reset()
        state = discretize_state(state)
        total_reward = 0
        env.render()
        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = discretize_state(next_state)
            state = next_state
            total_reward += reward
            env.render()
            if done:
                break
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")
    env.close()

def q_table_to_dataframe_combined(q_table, bins):
    data = []
    actions = range(q_table.shape[-1])
    state_indices = itertools.product(*[range(b) for b in bins])
    
    for state in state_indices:
        state_str = '-'.join(map(str, state))
        q_values = [q_table[state + (action,)] for action in actions]
        data.append((state_str, *q_values))
    
    columns = ['state'] + [f'q_value_action_{action}' for action in actions]
    df = pd.DataFrame(data, columns=columns)
    df_filtered = df[~((df['q_value_action_0'] == 0) & (df['q_value_action_1'] == 0))]
    return df_filtered

def save_q_table_to_excel_combined(q_table, bins, filename='q_table.xlsx'):
  
    df = q_table_to_dataframe_combined(q_table, bins)
    df.to_excel(filename, index=False)
    print(f"Q-table saved to {filename} with {len(df)} state-action pairs.")

if __name__ == "__main__":
    env = gymnasium.make('CartPole-v1')
    agent = QLearningAgent(env, NUM_BINS)

    print("Starting training...")
    training_rewards, moving_avg_rewards, first_five_q_tables, last_five_q_tables = train_agent(env, agent, episodes=10000)

    agent.save_as_json('q_table.json')
    save_q_table_to_excel_combined(agent.Q, NUM_BINS, filename='q_table_final.xlsx')

    plt.figure(figsize=(12, 6))
    plt.plot(training_rewards, label='Episode Reward', alpha=0.3)
    plt.plot(moving_avg_rewards, label='Moving Average (100 episodes)', color='orange')
    plt.axhline(y=200.0, color='red', linestyle='--', label='Solved Threshold')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards over Episodes')
    plt.legend()
    plt.grid(True)
    plt.show()

    os.makedirs('q_tables/first_five', exist_ok=True)
    os.makedirs('q_tables/last_five', exist_ok=True)

    for idx, q_table in enumerate(first_five_q_tables, start=1):
        json_filename = f'q_tables/first_five/q_table_episode_{idx}.json'
        excel_filename = f'q_tables/first_five/q_table_episode_{idx}.xlsx'
        save_q_table_to_excel_combined(q_table, NUM_BINS, filename=excel_filename)

    for idx, q_table in enumerate(last_five_q_tables, start=1):
        json_filename = f'q_tables/last_five/q_table_last_episode_{idx}.json'
        excel_filename = f'q_tables/last_five/q_table_last_episode_{idx}.xlsx'
        save_q_table_to_excel_combined(q_table, NUM_BINS, filename=excel_filename)

    print("\nQ-table snapshots saved for the first 5 and last 5 episodes as JSON and Excel files.")

    print("Evaluating agent...")
    evaluation_rewards = evaluate_agent(env, agent, episodes=100)

    plt.figure(figsize=(12, 6))
    plt.plot(evaluation_rewards, label='Evaluation Episode Reward')
    plt.axhline(y=200.0, color='red', linestyle='--', label='Solved Threshold')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Evaluation Rewards over Episodes')
    plt.legend()
    plt.grid(True)
    plt.show()

    env.close()
