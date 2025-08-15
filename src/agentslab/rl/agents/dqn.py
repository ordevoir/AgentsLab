from dataclasses import dataclass
from typing import Optional
import torch, torch.nn as nn
from agentslab.networks.q_network import QNetwork
from agentslab.utils.replay_memory import ReplayMemory, Transition

@dataclass
class DQNStats: loss: Optional[float]; epsilon: float; ep_return: float
class DQNAgent:
    def __init__(self, obs_dim, action_dim, hidden_sizes=(128,64), activation="relu",
                 gamma=0.99, batch_size=64, buffer_capacity=10000, lr=5e-4, weight_decay=1e-5, amsgrad=True,
                 double_dqn=True, soft_tau=0.005, device="cpu"):
        self.device=torch.device(device); self.gamma=gamma; self.batch_size=batch_size
        self.double_dqn=double_dqn; self.soft_tau=soft_tau; self.action_dim=action_dim
        self.online=QNetwork(obs_dim, action_dim, hidden_sizes, activation).to(self.device)
        self.target=QNetwork(obs_dim, action_dim, hidden_sizes, activation).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer=torch.optim.Adam(self.online.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=amsgrad)
        self.loss_fn=nn.SmoothL1Loss(); self.memory=ReplayMemory(buffer_capacity)
    def epsilon_greedy(self, state, epsilon: float)->int:
        if torch.rand(())>epsilon:
            with torch.no_grad(): return int(self.online(state).argmax(1).item())
        else: return int(torch.randint(0, self.action_dim, (1,)).item())
    def push(self, *args): self.memory.push(*args)
    def optimize(self)->Optional[float]:
        if len(self.memory)<self.batch_size: return None
        transitions=self.memory.sample(self.batch_size); batch=Transition(*zip(*transitions))
        mask=torch.tensor(tuple(s is not None for s in batch.next_state), device=self.device, dtype=torch.bool)
        non_final_next=torch.cat([s for s in batch.next_state if s is not None])
        state_b=torch.cat(batch.state); action_b=torch.cat(batch.action); reward_b=torch.cat(batch.reward)
        with torch.no_grad():
            q_next_online=self.online(non_final_next); q_next_target=self.target(non_final_next)
            best_next = q_next_online.argmax(1, keepdim=True) if self.double_dqn else q_next_target.argmax(1, keepdim=True)
            next_v=torch.zeros((self.batch_size,1), device=self.device); next_v[mask]=q_next_target.gather(1, best_next)
        q=self.online(state_b); q_sel=q.gather(1, action_b)
        target=reward_b.unsqueeze(1)+self.gamma*next_v
        loss=self.loss_fn(q_sel, target)
        self.optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_value_(self.online.parameters(), 5.); self.optimizer.step()
        if self.soft_tau and self.soft_tau>0:
            with torch.no_grad():
                for tp, sp in zip(self.target.parameters(), self.online.parameters()): tp.data.mul_(1-self.soft_tau).add_(self.soft_tau*sp.data)
        return float(loss.item())
    def hard_update(self): self.target.load_state_dict(self.online.state_dict())
