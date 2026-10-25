"""Deterministic checks for planning and model-free control algorithms."""

import gymnasium as gym
import numpy as np
import pytest

import icy_gridworld
from icy_gridworld.algorithms import policy_iteration, q_learning, sarsa, value_iteration
from icy_gridworld.dqn import dqn


def test_planning_algorithms_agree():
    env = icy_gridworld.IcyGridWorldEnv()
    value_result = value_iteration(env)
    policy_result = policy_iteration(env)

    assert value_result.policy.shape == (400,)
    assert value_result.q_values.shape == (400, 4)
    assert value_result.values[19] == 0
    assert policy_result.values == pytest.approx(value_result.values, abs=1e-7)
    assert policy_result.policy[380] == value_result.policy[380]
    assert value_result.iterations > 0
    assert policy_result.iterations > 0


@pytest.mark.parametrize("algorithm", [sarsa, q_learning])
def test_model_free_algorithms_update_q_values_reproducibly(algorithm):
    first = gym.make("IcyGridWorld-v0")
    second = gym.make("IcyGridWorld-v0")
    try:
        one = algorithm(first, episodes=20, seed=17)
        two = algorithm(second, episodes=20, seed=17)
    finally:
        first.close()
        second.close()

    assert one.policy.shape == (400,)
    assert one.q_values.shape == (400, 4)
    assert np.count_nonzero(one.q_values) > 0
    assert np.array_equal(one.q_values, two.q_values)
    assert one.episode_returns == two.episode_returns
    assert len(one.episode_lengths) == 20
    assert all(1 <= length <= 500 for length in one.episode_lengths)


@pytest.mark.parametrize("algorithm", [value_iteration, policy_iteration])
def test_planning_parameter_validation(algorithm):
    env = icy_gridworld.IcyGridWorldEnv()
    with pytest.raises(ValueError, match="gamma"):
        algorithm(env, gamma=1.0)


@pytest.mark.parametrize("algorithm", [sarsa, q_learning])
def test_learning_parameter_validation(algorithm):
    env = gym.make("IcyGridWorld-v0")
    try:
        with pytest.raises(ValueError, match="episodes"):
            algorithm(env, episodes=0)
        with pytest.raises(ValueError, match="alpha"):
            algorithm(env, alpha=0)
    finally:
        env.close()


def test_dqn_updates_network_reproducibly():
    settings = {
        "episodes": 3,
        "seed": 23,
        "max_steps": 50,
        "hidden_size": 8,
        "replay_capacity": 100,
        "batch_size": 8,
        "warmup_steps": 8,
        "train_frequency": 1,
        "target_update_frequency": 16,
    }
    first = gym.make("IcyGridWorld-v0")
    second = gym.make("IcyGridWorld-v0")
    try:
        one = dqn(first, **settings)
        two = dqn(second, **settings)
    finally:
        first.close()
        second.close()

    assert one.policy.shape == (400,)
    assert one.q_values.shape == (400, 4)
    assert np.isfinite(one.q_values).all()
    assert np.count_nonzero(one.q_values) > 0
    assert np.array_equal(one.q_values, two.q_values)
    assert one.episode_returns == two.episode_returns
    assert one.episode_lengths == (50, 50, 50)


def test_dqn_parameter_validation():
    env = gym.make("IcyGridWorld-v0")
    try:
        with pytest.raises(ValueError, match="episodes"):
            dqn(env, episodes=0)
        with pytest.raises(ValueError, match="learning_rate"):
            dqn(env, learning_rate=0)
        with pytest.raises(ValueError, match="replay"):
            dqn(env, replay_capacity=4, batch_size=8)
    finally:
        env.close()
