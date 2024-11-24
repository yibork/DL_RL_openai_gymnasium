import numpy as np
import gymnasium  # Ensure you're using Gymnasium
import random
import matplotlib.pyplot as plt
import json
import os

# 1. Discretization Function
NUM_BINS = (6, 6, 12, 12)
STATE_BOUNDS = list(zip(
    [-4.8, -5, -0.418, -5],
    [4.8, 5, 0.418, 5]
))

def discretize_state(state, bins=NUM_BINS, bounds=STATE_BOUNDS):
    """Convert a continuous state into a discrete state."""
    discretized = []
    for i in range(len(state)):
        if state[i] <= bounds[i][0]:
            bin_index = 0
        elif state[i] >= bounds[i][1]:
            bin_index = bins[i] - 1
        else:
            # Scale state to [0, 1] and then to [0, bins[i]-1]
            scale = (state[i] - bounds[i][0]) / (bounds[i][1] - bounds[i][0])
            bin_index = int(scale * bins[i])
            if bin_index >= bins[i]:
                bin_index = bins[i] - 1
        discretized.append(bin_index)
    return tuple(discretized)

# 2. Q-Learning Agent
class QLearningAgent:
    def __init__(self, env, bins, alpha=0.1, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.env = env
        self.bins = bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # Initialize Q-table with zeros
        self.Q = np.zeros(bins + (env.action_space.n,))

    def choose_action(self, state):
        """Choose an action using epsilon-greedy policy."""
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()  # Explore: random action
        else:
            return np.argmax(self.Q[state])  # Exploit: best action

    def learn(self, current_state, action, reward, next_state, done):
        """Update Q-table based on action taken."""
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state][best_next_action] * (not done)
        td_delta = td_target - self.Q[current_state][action]
        self.Q[current_state][action] += self.alpha * td_delta

        if done:
            # Decay exploration rate
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_as_json(self, filename='q_table.json'):
        """Save the Q-table to a JSON file."""
        q_table_list = self.Q.tolist()
        with open(filename, 'w') as f:
            json.dump(q_table_list, f)
        print(f"Q-table saved to {filename}.")

    def load_from_json(self, filename='q_table.json'):
        """Load the Q-table from a JSON file."""
        if not os.path.exists(filename):
            print(f"File {filename} not found. Starting with a fresh Q-table.")
            return
        with open(filename, 'r') as f:
            q_table_list = json.load(f)
        self.Q = np.array(q_table_list)
        print(f"Q-table loaded from {filename}.")

# 3. Helper Function to Save Q-Tables with Indices
def save_q_table(q_table, filename):
    """
    Save the Q-table to a JSON file.

    Parameters:
    - q_table (numpy.ndarray): The Q-table to save.
    - filename (str): The name of the file to save the Q-table.
    """
    q_table_list = q_table.tolist()
    with open(filename, 'w') as f:
        json.dump(q_table_list, f)
    print(f"Q-table saved to {filename}.")

# 4. Training Function
def train_agent(env, agent, episodes=10000, max_steps=200, window_size=100):
    rewards = []
    moving_avg_rewards = []
    first_five_q_tables = []  # To store Q-tables after episodes 1-5
    last_five_q_tables = []   # To store Q-tables for the last 5 episodes

    for episode in range(episodes):
        # Correctly unpack the reset function
        state, info = env.reset()
        state = discretize_state(state)
        total_reward = 0

        for step in range(max_steps):
            action = agent.choose_action(state)
            # Correctly unpack the step function
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = discretize_state(next_state)

            # Adjust reward for better learning
            if done and step < max_steps - 1:
                reward = -100

            agent.learn(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                break

        rewards.append(total_reward)

        # Capture Q-table after the first 5 episodes
        if episode < 5:
            first_five_q_tables.append(agent.Q.copy())

        # Maintain a queue for the last 5 episodes
        if len(last_five_q_tables) < 5:
            last_five_q_tables.append(agent.Q.copy())
        else:
            last_five_q_tables.pop(0)
            last_five_q_tables.append(agent.Q.copy())

        # Compute moving average
        if episode >= window_size:
            moving_avg = np.mean(rewards[-window_size:])
            moving_avg_rewards.append(moving_avg)
        else:
            moving_avg_rewards.append(np.mean(rewards))

        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            print(f"Episode {episode + 1}: Average Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}")

        # Early stopping
        if episode+1 == 1000:
            print(f"Solved after {episode + 1} episodes!")
            # Optionally, save the final Q-table
            agent.save_as_json('q_table_final.json')
            break

    return rewards, moving_avg_rewards, first_five_q_tables, last_five_q_tables

# 5. Evaluation Function
def evaluate_agent(env, agent, episodes=100, max_steps=200):
    total_rewards = []
    agent.epsilon = 0.0  # Disable exploration for evaluation

    for episode in range(episodes):
        # Correctly unpack the reset function
        state, info = env.reset()
        state = discretize_state(state)
        total_reward = 0

        for step in range(max_steps):
            action = agent.choose_action(state)
            # Correctly unpack the step function
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

# 6. Visualization Function (Optional)
def visualize_agent(env, agent, episodes=5, max_steps=200):
    for episode in range(episodes):
        state, info = env.reset()
        state = discretize_state(state)
        total_reward = 0
        env.render()  # Render the environment
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

# 7. Main Execution
if __name__ == "__main__":
    env = gymnasium.make('CartPole-v1')
    agent = QLearningAgent(env, NUM_BINS)

    print("Starting training...")
    training_rewards, moving_avg_rewards, first_five_q_tables, last_five_q_tables = train_agent(env, agent, episodes=10000)

    # Save the Q-table after training
    agent.save_as_json('q_table.json')

    # Plot training rewards and moving average
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

    # Define the helper function again (if not already defined above)
    def save_q_table(q_table, filename):
        """
        Save the Q-table to a JSON file.

        Parameters:
        - q_table (numpy.ndarray): The Q-table to save.
        - filename (str): The name of the file to save the Q-table.
        """
        q_table_list = q_table.tolist()
        with open(filename, 'w') as f:
            json.dump(q_table_list, f)
        print(f"Q-table saved to {filename}.")

    # Create directories to store Q-table snapshots if they don't exist
    os.makedirs('q_tables/first_five', exist_ok=True)
    os.makedirs('q_tables/last_five', exist_ok=True)

    # Save the Q-tables for the first 5 episodes
    for idx, q_table in enumerate(first_five_q_tables, start=1):
        filename = f'q_tables/first_five/q_table_episode_{idx}.json'
        save_q_table(q_table, filename)

    # Save the Q-tables for the last 5 episodes
    for idx, q_table in enumerate(last_five_q_tables, start=1):
        filename = f'q_tables/last_five/q_table_last_episode_{idx}.json'
        save_q_table(q_table, filename)

    print("\nQ-table snapshots saved for the first 5 and last 5 episodes.")

    print("Evaluating agent...")
    evaluation_rewards = evaluate_agent(env, agent, episodes=100)

    # Optionally, plot evaluation rewards
    plt.figure(figsize=(12, 6))
    plt.plot(evaluation_rewards, label='Evaluation Episode Reward')
    plt.axhline(y=200.0, color='red', linestyle='--', label='Solved Threshold')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Evaluation Rewards over Episodes')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Close the environment
    env.close()
