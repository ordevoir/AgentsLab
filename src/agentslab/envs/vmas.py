from __future__ import annotations
from torchrl.envs.libs.vmas import VmasEnv

def make_vmas_env(
    scenario: str = "navigation",
    num_envs: int = 32,
    device=None,
    vmas_device=None,
    continuous_actions: bool = True,
    max_steps: int | None = 200,
    seed: int | None = 0,
    **kwargs,
):
    return VmasEnv(
        scenario=scenario,
        num_envs=num_envs,
        continuous_actions=continuous_actions,
        max_steps=max_steps,
        seed=seed,
        device=vmas_device or device,
        **kwargs,
    )
