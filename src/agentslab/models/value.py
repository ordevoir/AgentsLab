from __future__ import annotations
from torchrl.objectives import DQNLoss, SoftUpdate

def make_dqn_loss(qvalue_actor, gamma: float, double_dqn: bool, delay_value: bool, loss_function: str):
    loss = DQNLoss(
        value_network=qvalue_actor,
        loss_function=loss_function,
        delay_value=delay_value,
        double_dqn=double_dqn,
    )
    loss.make_value_estimator(gamma=gamma)
    return loss

def make_soft_updater(loss_module, tau: float):
    return SoftUpdate(loss_module, tau=tau)
