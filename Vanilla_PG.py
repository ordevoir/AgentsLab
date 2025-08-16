from torch.distributions import Categorical
import numpy as np
import torch

# Policy
import torch.nn as nn

class Policy(nn.Module):
    def __init__(self, in_dim, out_dim, n_units=64) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, n_units),
            nn.ReLU(),
            nn.Linear(n_units, out_dim),
        )
        self.reset()
        self.train()    # метод суперкласса, устанавливает режим обучения

    def reset(self):
        self.log_probs = []
        self.rewards = []

    def forward(self, x):
        return self.model(x)
    
    def act(self, state):
        x = torch.from_numpy(state.astype(np.float32))
        logits = self.forward(x)
        rv = Categorical(logits=logits)
        action = rv.sample()
        self.log_probs.append(rv.log_prob(action))
        return action.item()    # <=> int(action)
    

# Trainer
def train(policy, optimizer, gamma):
    # ретроспективно посчитаем отдачи G_t на каждом шаге t эпизода
    T = len(policy.rewards)                 # длительность траектории
    rs = np.zeros(T, dtype=np.float32)      # retunrs
    future_return = 0.0
    for t in reversed(range(T)):            # в обратном порядке!
        reward = policy.rewards[t]
        future_return = reward + gamma * future_return
        rs[t] = future_return
    rs = torch.tensor(rs)
    rs = (rs - rs.mean()) / (rs.std() + 1e-8)  # нормализация отдач

    log_probs = torch.stack(policy.log_probs)
    loss = - torch.mean(log_probs * rs)      # минус для ascent вместо descent

    optimizer.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad_value_(policy.parameters(), 5.)
    optimizer.step()
    
    return loss.item()

# Environment

import gymnasium as gym

env = gym.make("CartPole-v1")
m = env.observation_space.shape[0]
n = env.action_space.n


# Training
policy = Policy(m, n, n_units=16)

gamma     = 0.99
episodes  = 600
max_steps = 500

learning_rate = 0.005
optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

total_rewards = []
for episode in range(episodes):
    obs, info = env.reset()
    for t in range(max_steps):
        action = policy.act(obs)
        obs, rew, term, trunc, info = env.step(action)
        policy.rewards.append(rew)
        if term or trunc:
            break
    loss = train(policy, optimizer, gamma)
    total_reward = sum(policy.rewards)
    policy.reset()
    print(f"\r{episode = }, {loss = :.1f}, {total_reward = }", end="")
    # соберем статистику
    total_rewards.append(total_reward)

# Evaluate

import numpy as np

def get_means(array, width=30):
    means = np.zeros_like(array, dtype=float)
    for i in range(len(means)):
        right = i + 1
        left = max(0, right-width)
        means[i] = array[left: right].mean()
    return means


import matplotlib.pyplot as plt

xs = np.arange(len(total_rewards))
rewards = np.array(total_rewards)
means = get_means(rewards, 30)

plt.plot(xs, rewards)
plt.plot(xs, means)

