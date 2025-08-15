import os, math, torch, gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf
import hydra
from agentslab.rl.agents.dqn import DQNAgent
from agentslab.utils.seed import set_seed

def _resolve_device(name:str)->str:
    if name=='auto':
        if torch.cuda.is_available(): return 'cuda'
        if getattr(torch.backends,'mps',None) and torch.backends.mps.is_available(): return 'mps'
        return 'cpu'
    return name
@hydra.main(config_path='../../configs', config_name='config_dqn', version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True)); set_seed(cfg.seed); device=_resolve_device(cfg.device)
    env=gym.make(cfg.rl.env.id, render_mode=None); obs,_=env.reset(seed=cfg.seed)
    obs_dim=env.observation_space.shape[0]; action_dim=env.action_space.n
    agent=DQNAgent(obs_dim, action_dim, cfg.rl.model.hidden_sizes, cfg.rl.model.activation, cfg.rl.training.gamma,
                   cfg.rl.training.batch_size, cfg.rl.training.buffer_capacity, cfg.rl.optimizer.lr, cfg.rl.optimizer.weight_decay,
                   getattr(cfg.rl.optimizer,'amsgrad',True), cfg.rl.training.double_dqn, cfg.rl.training.soft_update_tau, device)
    writer=SummaryWriter(log_dir=os.path.join(os.getcwd(),'tb')) if cfg.common.use_tensorboard else None
    eps_start,eps_end,eps_decay=cfg.rl.training.eps_start,cfg.rl.training.eps_end,cfg.rl.training.eps_decay
    hard_every=cfg.rl.training.hard_update_every
    log_every,save_every=cfg.rl.training.log_every,cfg.rl.training.save_every
    max_episodes=cfg.rl.training.max_episodes
    max_steps=cfg.rl.training.max_steps_per_episode
    global_steps=0
    for ep in range(1, max_episodes+1):
        obs,_=env.reset(); state=torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
        ep_return=0.0; loss_val=None; steps_in_ep=0
        while True:
            epsilon=eps_end+(eps_start-eps_end)*math.exp(-1.0*global_steps/eps_decay)
            action=agent.epsilon_greedy(state, epsilon)
            obs, rew, term, trunc, _ = env.step(action)
            reward=torch.tensor([rew], device=agent.device)
            next_state=None if term else torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
            agent.push(state, torch.tensor([[action]], device=agent.device), next_state, reward)
            state = next_state if next_state is not None else torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
            ep_return += float(rew); global_steps += 1; steps_in_ep += 1
            loss_val=agent.optimize()
            if hard_every and hard_every>0 and (global_steps%hard_every==0): agent.hard_update()
            if term or trunc: break
            if max_steps and steps_in_ep>=max_steps: break
        if writer: 
            writer.add_scalar('charts/ep_return', ep_return, ep)
            writer.add_scalar('policy/epsilon', epsilon, ep)
            if loss_val is not None: writer.add_scalar('loss/td_loss', loss_val, ep)
        if ep%log_every==0: print(f"[ep {ep:04d}] return={ep_return:.2f} eps={epsilon:.3f} loss={None if loss_val is None else f'{loss_val:.4f}'}")
        if save_every and (ep%save_every==0 or ep==max_episodes):
            ckpt=os.path.join(os.getcwd(),'checkpoints','rl'); os.makedirs(ckpt, exist_ok=True)
            torch.save(agent.online.state_dict(), os.path.join(ckpt, f'dqn_online_ep{ep}.pt'))
            torch.save(agent.target.state_dict(), os.path.join(ckpt, f'dqn_target_ep{ep}.pt'))
    env.close()

if __name__=='__main__': 
    main()
