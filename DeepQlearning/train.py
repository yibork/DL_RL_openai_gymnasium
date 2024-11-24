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

STATE_BOUNDS = list(zip(
    [-4.8, -5, -0.418, -5],
    [4.8, 5, 0.418, 5]
))

def normalize_state(state, bounds=STATE_BOUNDS):
    normalized = []
    for i in range(len(state)):
        min_val, max_val = bounds[i]
        state_val = np.clip(state[i], min_val, max_val)
        normalized_val = (state_val - min_val) / (max_val - min_val)
        normalized.append(normalized_val)
    return np.array(normalized)

class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
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
        return len(self.buffer)

class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        lr=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=500,
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
        self.policy_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)
    
    def select_action(self, state):
        self.steps_done += 1
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
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device).unsqueeze(1)
        rewards = rewards.to(self.device).unsqueeze(1)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device).unsqueeze(1)
        
        current_q = self.policy_net(states).gather(1, actions)
        
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (self.gamma * max_next_q * (1 - dones))
        
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, path='dqn_model_final.pth'):
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}.")
    
    def load_model(self, path='dqn_model_final.pth'):
        if os.path.exists(path):
            self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"Model loaded from {path}.")
        else:
            print(f"No model found at {path}. Starting with a fresh model.")

def train_dqn_agent(
    env,
    agent,
    num_episodes=1000,          
    max_steps=200,
    target_update_interval=1000,
    log_interval=100
):
    rewards = []
    moving_avg_rewards = []
    total_steps = 0
    
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        state = normalize_state(state)
        episode_reward = 0
    
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = normalize_state(next_state)
            episode_reward += reward
    
            if done and step < max_steps - 1:
                reward = -100.0
    
            agent.store_experience(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            total_steps += 1
    
            if total_steps % target_update_interval == 0:
                agent.update_target_network()
                print(f"Target network updated at step {total_steps}.")
    
            if done:
                break
    
        rewards.append(episode_reward)
        moving_avg = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
        moving_avg_rewards.append(moving_avg)
    
        if episode % log_interval == 0:
            print(f"Episode {episode}: Average Reward: {moving_avg:.2f}, Epsilon: {agent.epsilon:.4f}")
    
    return rewards, moving_avg_rewards

def evaluate_dqn_agent(env, agent, num_episodes=100, max_steps=200):
    total_rewards = []
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        state = normalize_state(state)
        episode_reward = 0
    
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = normalize_state(next_state)
            episode_reward += reward
            state = next_state
    
            if done:
                break
    
        total_rewards.append(episode_reward)
    
    agent.epsilon = original_epsilon
    avg_reward = np.mean(total_rewards)
    print(f"Average Reward over {num_episodes} Evaluation Episodes: {avg_reward:.2f}")
    return total_rewards

def plot_rewards(rewards, moving_avg_rewards, title='DQN Training Rewards', save_path='training_rewards.png'):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Episode Reward', alpha=0.3)
    plt.plot(moving_avg_rewards, label='Moving Average (100 episodes)', color='orange')
    plt.axhline(y=195.0, color='red', linestyle='--', label='Solved Threshold')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Training rewards plot saved to {save_path}.")

def plot_evaluation_rewards(evaluation_rewards, title='DQN Evaluation Rewards', save_path='evaluation_rewards.png'):
    plt.figure(figsize=(12, 6))
    plt.plot(evaluation_rewards, label='Evaluation Episode Reward')
    plt.axhline(y=195.0, color='red', linestyle='--', label='Solved Threshold')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Evaluation rewards plot saved to {save_path}.")

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        lr=0.001,                   
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=500,
        replay_capacity=10000,
        batch_size=64,
        target_update_freq=1000
    )
    print("Starting DQN training...")
    training_rewards, moving_avg_rewards = train_dqn_agent(
        env=env,
        agent=agent,
        num_episodes=1000,          
        max_steps=200,
        target_update_interval=1000,
        log_interval=100
    )
    plot_rewards(training_rewards, moving_avg_rewards, title='DQN Training Rewards', save_path='training_rewards.png')
    agent.save_model('dqn_model_final.pth')
    print("Evaluating DQN agent...")
    evaluation_rewards = evaluate_dqn_agent(env, agent, num_episodes=100, max_steps=200)
    plot_evaluation_rewards(evaluation_rewards, title='DQN Evaluation Rewards', save_path='evaluation_rewards.png')
    env.close()
