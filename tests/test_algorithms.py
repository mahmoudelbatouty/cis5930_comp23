"""Deterministic checks for planning and model-free control algorithms."""

import gymnasium as gym
import numpy as np
import pytest

import icy_gridworld
from icy_gridworld.algorithms import policy_iteration, q_learning, sarsa, value_iteration


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
