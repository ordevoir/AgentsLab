
# Replay Buffer

from collections import namedtuple, deque
import random

Transition = namedtuple("Transition", "state action next_state reward")

class ReplayMemory(object):
    def __init__(self, capasity) -> None:
        self.__memory = deque([], maxlen=capasity)
    
    def push(self, *args):
        self.__memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.__memory, batch_size)
    
    def __len__(self):
        return len(self.__memory)
    
    def clear(self):
        self.__memory.clear()


# Device
import torch

device = (
         "cuda" if torch.cuda.is_available()
    else "mps"  if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Environment

import gymnasium as gym

env = gym.make("CartPole-v1")
m = env.observation_space.shape[0]
n = env.action_space.n


# Deep Q-Networks

import torch.nn as nn 
from torch import tensor


class DQN(nn.Module):
    def __init__(self, n_obs, n_act):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(n_obs, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_act),
        )
    def forward(self, x):
        return self.stack(x)
    
online_net = DQN(m, n).to(device)
target_net = DQN(m, n).to(device)
target_net.load_state_dict(online_net.state_dict())

# ε-greedy policy

def epsilon_greedy_policy(model, state, env, epsilon=0):
    sample = random.random()
    if sample > epsilon:
        with torch.no_grad():
            Q_values = model(state)
        action = Q_values.max(dim=1).indices    # индекс максимального элемента
        return action.view(1, 1)
    else:
        action = env.action_space.sample()
        return tensor([[action]], device=device, dtype=torch.long)


# Trainer

import torch.optim as optim
optimizer = optim.Adam(online_net.parameters(), lr=LR, amsgrad=True,
                       weight_decay=1e-5,
                       )
loss_fn = nn.SmoothL1Loss()
memory = ReplayMemory(10000)


BATCH_SIZE = 64
GAMMA = 0.99
ALPHA = 0.005
LR = 5e-4
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 2000

def optimize_model(online_net, target_net, memory, loss_fn):
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))      # транспонирование transitions
    
    mask = tuple(map(lambda s: s is not None, batch.next_state))
    mask = tensor(mask, device=device, dtype=torch.bool)
    non_term_next_states = [s for s in batch.next_state if s is not None]
    non_term_next_states = torch.cat(non_term_next_states)
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    with torch.no_grad():
        next_Q_online = online_net(non_term_next_states)
        next_Q_target = target_net(non_term_next_states)

    # best_next_actions = next_Q_target.max(1).indices.unsqueeze(1)
    best_next_actions = next_Q_online.max(1).indices.unsqueeze(1)
    next_V = torch.zeros((BATCH_SIZE, 1), device=device)
    next_V[mask] = next_Q_target.gather(1, best_next_actions)
    Q_targets = reward_batch.unsqueeze(1) + GAMMA * next_V      

    Q = online_net(state_batch)
    Q_of_selected_actions = Q.gather(dim=1, index=action_batch)

    loss = loss_fn(Q_of_selected_actions, Q_targets)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(online_net.parameters(), 5.)
    optimizer.step()

# Training

from itertools import count
import math
n_episodes = 300
env = gym.make("CartPole-v1", render_mode=None)
steps = 0

rewards = []
epsilons = []

for episode in range(n_episodes):
    obs, info = env.reset()
    state = tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        # epsilon = max(1 - episode / n_episodes, 0.01)
        epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps / EPS_DECAY)
        steps += 1

        action = epsilon_greedy_policy(online_net, state, env, epsilon)
        obs, rew, term, trunc, info = env.step(action.item())
        reward = tensor([rew], device=device)

        if term:
            next_state = None
        else:
            next_state = tensor(obs, dtype=torch.float32, device=device)
            next_state.unsqueeze_(0)

        memory.push(state, action, next_state, reward)
        state = next_state
        optimize_model(online_net, target_net, memory, loss_fn)
        
        target_net_state_dict = target_net.state_dict()
        online_net_state_dict = online_net.state_dict()

        for key in online_net_state_dict:
            target_net_state_dict[key] = ALPHA  * online_net_state_dict[key] + \
                                    (1 - ALPHA) * target_net_state_dict[key]
        target_net.load_state_dict(target_net_state_dict)
        if term or trunc:
            break
    print(f"\rEpisode: {episode+1}, Steps: {t+1}, eps: {epsilon:.3f}", end="")
    rewards.append(t+1)
    epsilons.append(epsilon)

env.close()

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

xs = np.arange(len(rewards))
rewards = np.array(rewards)
means = get_means(rewards, 30)

plt.plot(xs, rewards)
plt.plot(xs, means)

def play(env, policy):
    policy.eval()
    obs, _ = env.reset()
    total = 0.0
    with torch.no_grad():
        for _ in range(200):
            s = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            a = policy(s).argmax(1).item()
            obs, r, term, trunc, _ = env.step(a)
            total += r
            if term or trunc:
                break
    env.close()
    return total

env = gym.make("CartPole-v1", render_mode="human")
play(env, target_net)