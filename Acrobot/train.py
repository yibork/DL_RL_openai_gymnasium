import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================
# 1. State Normalization
# ==========================

STATE_BOUNDS = list(zip(
    [-1.0, -1.0, -1.0, -1.0, -12.0, -12.0],  # Minimum bounds
    [1.0, 1.0, 1.0, 1.0, 12.0, 12.0]          # Maximum bounds
))

def normalize_state(state, bounds=STATE_BOUNDS):
    """
    Normalize the state to [0, 1] based on predefined bounds.
    """
    normalized = []
    for i in range(len(state)):
        min_val, max_val = bounds[i]
        state_val = np.clip(state[i], min_val, max_val)
        normalized_val = (state_val - min_val) / (max_val - min_val)
        normalized.append(normalized_val)
    return np.array(normalized)

# ==========================
# 2. Neural Network Architecture
# ==========================

class DQNNetwork(nn.Module):
    """
    Deep Q-Network (DQN) Architecture.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        """
        Forward pass through the network.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

# ==========================
# 3. Replay Buffer
# ==========================

class ReplayBuffer:
    """
    Experience Replay Buffer.
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        Store an experience in the buffer.
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        """
        Return the current size of internal memory.
        """
        return len(self.buffer)

# ==========================
# 4. DQN Agent
# ==========================

class DQNAgent:
    """
    Deep Q-Network Agent.
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        lr=0.001,                  
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=5000,  # Increased decay rate
        replay_capacity=10000,
        batch_size=64,
        target_update_freq=1000
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps_done = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize policy and target networks
        self.policy_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer and loss function
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr) 
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)
    
    def select_action(self, state):
        """
        Select an action using epsilon-greedy policy.
        """
        self.steps_done += 1
        # Epsilon decay
        self.epsilon = self.epsilon_min + (1.0 - self.epsilon_min) * \
            np.exp(-1. * self.steps_done / self.epsilon_decay)
        
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """
        Store an experience in the replay buffer.
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self):
        """
        Update the policy network using a batch of experiences.
        """
        if len(self.replay_buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device).unsqueeze(1)
        rewards = rewards.to(self.device).unsqueeze(1)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device).unsqueeze(1)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions)
        
        # Compute target Q values
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (self.gamma * max_next_q * (1 - dones))
        
        # Compute loss
        loss = self.loss_fn(current_q, target_q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        """
        Update the target network to match the policy network.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, path='dqn_model.pth'):
        """
        Save the policy network's state_dict.
        """
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}.")
    
    def load_model(self, path='dqn_model.pth'):
        """
        Load the policy network's state_dict.
        """
        if os.path.exists(path):
            self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"Model loaded from {path}.")
        else:
            print(f"No model found at {path}. Starting with a fresh model.")

# ==========================
# 5. Training Function
# ==========================

def train_dqn_agent(
    env,
    agent,
    num_episodes=1000,          
    max_steps=500,  # Increased max_steps for Acrobot-v1
    target_update_interval=1000,
    log_interval=100
):
    """
    Train the DQN agent in the given environment.
    """
    rewards = []
    moving_avg_rewards = []
    total_steps = 0

    for episode in range(1, num_episodes + 1):
        state, info = env.reset()
        state = normalize_state(state)
        episode_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = normalize_state(next_state)
            episode_reward += reward

            # No additional reward adjustment needed for Acrobot-v1

            agent.store_experience(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            total_steps += 1

            # Update target network periodically
            if total_steps % target_update_interval == 0:
                agent.update_target_network()
                print(f"Target network updated at step {total_steps}.")

            if done:
                break

        rewards.append(episode_reward)
        moving_avg = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
        moving_avg_rewards.append(moving_avg)

        # Logging
        if episode % log_interval == 0:
            print(f"Episode {episode}: Total Reward: {episode_reward:.2f}, "
                  f"Average Reward: {moving_avg:.2f}, Epsilon: {agent.epsilon:.4f}")

    return rewards, moving_avg_rewards

# ==========================
# 6. Evaluation Function
# ==========================

def evaluate_dqn_agent(env, agent, num_episodes=100, max_steps=500):
    """
    Evaluate the trained DQN agent.
    """
    total_rewards = []
    agent.epsilon = 0.0  # Disable exploration

    for episode in range(1, num_episodes + 1):
        state, info = env.reset()
        state = normalize_state(state)
        episode_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = normalize_state(next_state)
            episode_reward += reward
            state = next_state

            if done:
                break

        total_rewards.append(episode_reward)

    avg_reward = np.mean(total_rewards)
    print(f"Average Reward over {num_episodes} Evaluation Episodes: {avg_reward:.2f}")
    return total_rewards

# ==========================
# 7. Visualization Functions
# ==========================

def plot_rewards(rewards, moving_avg_rewards, title='DQN Training Rewards'):
    """
    Plot the training rewards and moving average rewards.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Episode Reward', alpha=0.3)
    plt.plot(moving_avg_rewards, label='Moving Average (100 episodes)', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_evaluation_rewards(evaluation_rewards, title='DQN Evaluation Rewards'):
    """
    Plot the evaluation rewards.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(evaluation_rewards, label='Evaluation Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# ==========================
# 8. Main Execution
# ==========================

if __name__ == "__main__":
    # Initialize environment
    env = gym.make('Acrobot-v1')

    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]  # 6 for Acrobot
    action_dim = env.action_space.n             # 3 for Acrobot

    # Initialize agent with specified hyperparameters
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        lr=0.001,                  
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=5000,  # Increased decay rate
        replay_capacity=10000,
        batch_size=64,
        target_update_freq=1000
    )

    # Optionally, load a pre-trained model
    # agent.load_model('dqn_model_final.pth')

    print("Starting DQN training on Acrobot-v1...")
    training_rewards, moving_avg_rewards = train_dqn_agent(
        env=env,
        agent=agent,
        num_episodes=1000,         
        max_steps=500,  # Increased max_steps
        target_update_interval=1000,
        log_interval=100
    )

    # Plot training rewards
    plot_rewards(training_rewards, moving_avg_rewards, title='DQN Training Rewards on Acrobot-v1')

    # Save the final model
    agent.save_model('dqn_model_final.pth')

    # Evaluate the trained agent
    print("Evaluating DQN agent on Acrobot-v1...")
    evaluation_rewards = evaluate_dqn_agent(env, agent, num_episodes=100, max_steps=500)

    # Plot evaluation rewards
    plot_evaluation_rewards(evaluation_rewards, title='DQN Evaluation Rewards on Acrobot-v1')

    # Close the environment
    env.close()
