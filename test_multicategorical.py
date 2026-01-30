"""Tests for MultiCategorical distribution."""
import torch
import sys

def test_multicategorical():
    print('=' * 60)
    print('Testing MultiCategorical')
    print('=' * 60)

    from agentslab.policies import MultiCategorical

    # Test 1: Basic construction with logits
    print('\n[Test 1] Construction with logits')
    nvec = [3, 5, 4]
    batch_size = 8
    logits = torch.randn(batch_size, sum(nvec))
    dist = MultiCategorical(nvec, logits=logits)
    print(f'  nvec: {nvec}')
    print(f'  logits shape: {logits.shape}')
    print(f'  dist: {dist}')
    print(f'  batch_shape: {dist.batch_shape}')
    print(f'  event_shape: {dist.event_shape}')
    print('  [OK]')

    # Test 2: Sample
    print('\n[Test 2] sample()')
    samples = dist.sample()
    print(f'  samples shape: {samples.shape}')
    print(f'  samples dtype: {samples.dtype}')
    print(f'  samples[0]: {samples[0].tolist()}')
    for i, n in enumerate(nvec):
        assert samples[..., i].min() >= 0, f'Sample {i} below 0'
        assert samples[..., i].max() < n, f'Sample {i} >= {n}'
    print('  All samples within valid ranges')
    print('  [OK]')

    # Test 3: Sample with sample_shape
    print('\n[Test 3] sample(sample_shape)')
    samples_shaped = dist.sample((2, 3))
    print(f'  sample_shape: (2, 3)')
    print(f'  result shape: {samples_shaped.shape}')
    assert samples_shaped.shape == (2, 3, batch_size, len(nvec))
    print('  [OK]')

    # Test 4: log_prob
    print('\n[Test 4] log_prob()')
    log_p = dist.log_prob(samples)
    print(f'  log_prob shape: {log_p.shape}')
    print(f'  log_prob values: {log_p[:3].tolist()}')
    assert log_p.shape == (batch_size,), f'Expected ({batch_size},), got {log_p.shape}'
    assert (log_p <= 0).all(), 'log_prob should be <= 0'
    print('  [OK]')

    # Test 5: entropy
    print('\n[Test 5] entropy()')
    ent = dist.entropy()
    print(f'  entropy shape: {ent.shape}')
    print(f'  entropy values: {ent[:3].tolist()}')
    assert ent.shape == (batch_size,)
    assert (ent >= 0).all(), 'Entropy should be >= 0'
    print('  [OK]')

    # Test 6: mode
    print('\n[Test 6] mode property')
    mode = dist.mode
    print(f'  mode shape: {mode.shape}')
    print(f'  mode[0]: {mode[0].tolist()}')
    assert mode.shape == (batch_size, len(nvec))
    print('  [OK]')

    # Test 7: Construction with probs
    print('\n[Test 7] Construction with probs')
    probs_list = [torch.softmax(torch.randn(batch_size, n), dim=-1) for n in nvec]
    probs = torch.cat(probs_list, dim=-1)
    dist_probs = MultiCategorical(nvec, probs=probs)
    samples_probs = dist_probs.sample()
    log_p_probs = dist_probs.log_prob(samples_probs)
    print(f'  probs shape: {probs.shape}')
    print(f'  samples shape: {samples_probs.shape}')
    print(f'  log_prob shape: {log_p_probs.shape}')
    print('  [OK]')

    # Test 8: Verify log_prob is sum of individual log_probs
    print('\n[Test 8] Verify log_prob = sum of individual log_probs')
    from torch.distributions import Categorical
    offset = 0
    manual_log_prob = torch.zeros(batch_size)
    for i, n in enumerate(nvec):
        cat = Categorical(logits=logits[..., offset:offset+n])
        manual_log_prob += cat.log_prob(samples[..., i])
        offset += n
    diff = (log_p - manual_log_prob).abs().max().item()
    print(f'  Max difference: {diff:.2e}')
    assert diff < 1e-5, f'log_prob mismatch: {diff}'
    print('  [OK]')

    # Test 9: Verify entropy is sum of individual entropies
    print('\n[Test 9] Verify entropy = sum of individual entropies')
    offset = 0
    manual_entropy = torch.zeros(batch_size)
    for i, n in enumerate(nvec):
        cat = Categorical(logits=logits[..., offset:offset+n])
        manual_entropy += cat.entropy()
        offset += n
    diff = (ent - manual_entropy).abs().max().item()
    print(f'  Max difference: {diff:.2e}')
    assert diff < 1e-5, f'entropy mismatch: {diff}'
    print('  [OK]')

    # Test 10: Error handling - both logits and probs
    print('\n[Test 10] Error handling - both logits and probs')
    try:
        MultiCategorical(nvec, logits=logits, probs=probs)
        print('  Should have raised ValueError!')
        return False
    except ValueError as e:
        print(f'  Caught expected error: {e}')
    print('  [OK]')

    # Test 11: Error handling - wrong logits size
    print('\n[Test 11] Wrong logits size')
    try:
        MultiCategorical(nvec, logits=torch.randn(batch_size, 5))
        print('  Should have raised ValueError!')
        return False
    except ValueError as e:
        print(f'  Caught expected error: {e}')
    print('  [OK]')

    # Test 12: Single sub-action (edge case)
    print('\n[Test 12] Single sub-action')
    dist_single = MultiCategorical([5], logits=torch.randn(4, 5))
    s = dist_single.sample()
    lp = dist_single.log_prob(s)
    print(f'  nvec: [5], samples shape: {s.shape}, log_prob shape: {lp.shape}')
    print('  [OK]')

    # Test 13: Large nvec
    print('\n[Test 13] Large nvec')
    large_nvec = [10, 20, 15, 8, 12]
    large_logits = torch.randn(16, sum(large_nvec))
    dist_large = MultiCategorical(large_nvec, logits=large_logits)
    s_large = dist_large.sample()
    lp_large = dist_large.log_prob(s_large)
    print(f'  nvec: {large_nvec}, sum: {sum(large_nvec)}')
    print(f'  samples shape: {s_large.shape}')
    print(f'  log_prob shape: {lp_large.shape}')
    print('  [OK]')

    print('\n' + '=' * 60)
    print('All MultiCategorical tests passed!')
    print('=' * 60)
    return True


def test_build_stochastic_actor_multidiscrete():
    print('\n' + '=' * 60)
    print('Testing build_stochastic_actor with MultiDiscrete')
    print('=' * 60)

    import torch.nn as nn
    from torchrl.data.tensor_specs import MultiDiscreteTensorSpec
    from agentslab.policies import get_num_action_logits, build_stochastic_actor

    # Create MultiDiscrete action spec
    nvec = [3, 5, 4]
    action_spec = MultiDiscreteTensorSpec(nvec=nvec)
    print(f'\n[Test 1] Action spec: {action_spec}')
    print(f'  nvec: {nvec}')

    # Get number of logits
    num_logits = get_num_action_logits(action_spec)
    print(f'  num_logits: {num_logits}')
    assert num_logits == sum(nvec), f'Expected {sum(nvec)}, got {num_logits}'
    print('  [OK]')

    # Create network
    print('\n[Test 2] Build actor')
    obs_dim = 10
    network = nn.Sequential(
        nn.Linear(obs_dim, 32),
        nn.ReLU(),
        nn.Linear(32, num_logits),
    )

    actor = build_stochastic_actor(network, action_spec, return_log_prob=True)
    print(f'  actor: {type(actor).__name__}')
    print('  [OK]')

    # Test forward pass with TensorDict
    print('\n[Test 3] Forward pass')
    from tensordict import TensorDict

    batch_size = 4
    td = TensorDict({
        "observation": torch.randn(batch_size, obs_dim)
    }, batch_size=[batch_size])

    td_out = actor(td)
    print(f'  Input keys: {list(td.keys())}')
    print(f'  Output keys: {list(td_out.keys())}')

    assert "action" in td_out.keys(), "action not in output"
    # TorchRL uses "action_log_prob" or "sample_log_prob" depending on version
    log_prob_key = "action_log_prob" if "action_log_prob" in td_out.keys() else "sample_log_prob"
    assert log_prob_key in td_out.keys(), f"neither action_log_prob nor sample_log_prob in output: {list(td_out.keys())}"

    action = td_out["action"]
    log_prob = td_out[log_prob_key]

    print(f'  action shape: {action.shape}')
    print(f'  action[0]: {action[0].tolist()}')
    print(f'  log_prob shape: {log_prob.shape}')
    print(f'  log_prob[0]: {log_prob[0].item():.4f}')

    assert action.shape == (batch_size, len(nvec)), f'Wrong action shape: {action.shape}'
    assert log_prob.shape == (batch_size,), f'Wrong log_prob shape: {log_prob.shape}'
    print('  [OK]')

    # Test that actions are within valid ranges
    print('\n[Test 4] Action ranges')
    for i, n in enumerate(nvec):
        assert action[..., i].min() >= 0, f'Action {i} below 0'
        assert action[..., i].max() < n, f'Action {i} >= {n}'
        print(f'  action[{i}]: 0 <= values < {n}')
    print('  [OK]')

    print('\n' + '=' * 60)
    print('All build_stochastic_actor tests passed!')
    print('=' * 60)
    return True


if __name__ == "__main__":
    success = True

    try:
        success = test_multicategorical() and success
    except Exception as e:
        print(f'\nERROR in test_multicategorical: {e}')
        import traceback
        traceback.print_exc()
        success = False

    try:
        success = test_build_stochastic_actor_multidiscrete() and success
    except Exception as e:
        print(f'\nERROR in test_build_stochastic_actor: {e}')
        import traceback
        traceback.print_exc()
        success = False

    print('\n' + '=' * 60)
    if success:
        print('ALL TESTS PASSED!')
    else:
        print('SOME TESTS FAILED!')
    print('=' * 60)

    sys.exit(0 if success else 1)
