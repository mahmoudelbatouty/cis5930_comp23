"""Planning and model-free control algorithms for Icy Grid World."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AlgorithmResult:
    """Policy and diagnostics produced by a planning or learning algorithm."""

    policy: np.ndarray
    values: np.ndarray
    q_values: np.ndarray
    iterations: int
    episode_returns: tuple[float, ...] = ()
    episode_lengths: tuple[int, ...] = ()


def _validate_hyperparameters(gamma, tolerance=None, max_iterations=None):
    if not 0 <= gamma < 1:
        raise ValueError("gamma must be in [0, 1)")
    if tolerance is not None and tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if max_iterations is not None and max_iterations < 1:
        raise ValueError("max_iterations must be positive")


def _states(env):
    return np.array(
        [state for state in range(env.observation_space.n)
         if divmod(state, 20) not in env.obstacles],
        dtype=int,
    )


def _model(env, states):
    """Materialize the sparse transition model in small dense arrays."""
    successors = np.zeros((env.observation_space.n, env.action_space.n, 3), dtype=int)
    probabilities = np.zeros_like(successors, dtype=float)
    rewards = np.zeros_like(successors, dtype=float)
    goal_state = 20 * env.goal[0] + env.goal[1]
    for state in states:
        for action in range(env.action_space.n):
            outcomes = env.transition_probabilities(state, action).items()
            for index, (next_state, probability) in enumerate(outcomes):
                successors[state, action, index] = next_state
                probabilities[state, action, index] = probability
                if state != goal_state:
                    rewards[state, action, index] = (
                        100.0 if next_state == goal_state else -1.0
                    )
    return successors, probabilities, rewards


def _action_values(values, gamma, model):
    successors, probabilities, rewards = model
    return (probabilities * (rewards + gamma * values[successors])).sum(axis=2)


def value_iteration(env, gamma=0.99, tolerance=1e-10, max_iterations=100_000):
    """Solve the infinite-horizon discounted MDP with value iteration."""
    _validate_hyperparameters(gamma, tolerance, max_iterations)
    states = _states(env)
    model = _model(env, states)
    values = np.zeros(env.observation_space.n, dtype=float)

    for iteration in range(1, max_iterations + 1):
        q_values = _action_values(values, gamma, model)
        updated = q_values.max(axis=1)
        if np.max(np.abs(updated[states] - values[states])) < tolerance:
            values = updated
            q_values = _action_values(values, gamma, model)
            return AlgorithmResult(
                q_values.argmax(axis=1), values, q_values, iteration
            )
        values = updated
    raise RuntimeError("value iteration failed to converge")


def policy_iteration(env, gamma=0.99, tolerance=1e-10, max_iterations=10_000):
    """Solve the exact MDP by alternating policy evaluation and improvement."""
    _validate_hyperparameters(gamma, tolerance, max_iterations)
    states = _states(env)
    model = _model(env, states)
    goal_state = 20 * env.goal[0] + env.goal[1]
    decision_states = states[states != goal_state]
    policy = np.zeros(env.observation_space.n, dtype=int)
    values = np.zeros(env.observation_space.n, dtype=float)

    for iteration in range(1, max_iterations + 1):
        for _ in range(max_iterations):
            q_values = _action_values(values, gamma, model)
            updated = values.copy()
            updated[decision_states] = q_values[
                decision_states, policy[decision_states]
            ]
            if np.max(np.abs(updated[states] - values[states])) < tolerance:
                values = updated
                break
            values = updated
        else:
            raise RuntimeError("policy evaluation failed to converge")

        q_values = _action_values(values, gamma, model)
        improved = q_values.argmax(axis=1)
        if np.array_equal(improved[decision_states], policy[decision_states]):
            return AlgorithmResult(policy.copy(), values, q_values, iteration)
        policy[decision_states] = improved[decision_states]
    raise RuntimeError("policy iteration failed to converge")


def _validate_learning_parameters(
    episodes, alpha, gamma, epsilon, epsilon_decay, min_epsilon, max_steps
):
    _validate_hyperparameters(gamma)
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    if not 0 <= min_epsilon <= epsilon <= 1:
        raise ValueError("epsilon values must satisfy 0 <= min_epsilon <= epsilon <= 1")
    if not 0 < epsilon_decay <= 1:
        raise ValueError("epsilon_decay must be in (0, 1]")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")


def _epsilon_greedy(q_values, state, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(q_values.shape[1]))
    best = np.flatnonzero(q_values[state] == q_values[state].max())
    return int(rng.choice(best))


def _learning_result(q_values, returns, lengths, episodes):
    return AlgorithmResult(
        policy=q_values.argmax(axis=1),
        values=q_values.max(axis=1),
        q_values=q_values,
        iterations=episodes,
        episode_returns=tuple(returns),
        episode_lengths=tuple(lengths),
    )


def q_learning(
    env,
    episodes=3_000,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.997,
    min_epsilon=0.05,
    seed=42,
    max_steps=500,
):
    """Learn an off-policy control policy using one-step Q-learning."""
    _validate_learning_parameters(
        episodes, alpha, gamma, epsilon, epsilon_decay, min_epsilon, max_steps
    )
    rng = np.random.default_rng(seed)
    q_values = np.zeros((env.observation_space.n, env.action_space.n), dtype=float)
    returns, lengths = [], []
    current_epsilon = epsilon

    for episode in range(episodes):
        state, _ = env.reset(seed=seed if episode == 0 else None)
        total_reward = 0.0
        for step in range(1, max_steps + 1):
            action = _epsilon_greedy(q_values, state, current_epsilon, rng)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated or step == max_steps
            target = reward if done else reward + gamma * q_values[next_state].max()
            q_values[state, action] += alpha * (target - q_values[state, action])
            total_reward += reward
            state = next_state
            if done:
                break
        returns.append(total_reward)
        lengths.append(step)
        current_epsilon = max(min_epsilon, current_epsilon * epsilon_decay)

    return _learning_result(q_values, returns, lengths, episodes)


def sarsa(
    env,
    episodes=3_000,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.997,
    min_epsilon=0.05,
    seed=42,
    max_steps=500,
):
    """Learn an on-policy control policy using one-step SARSA."""
    _validate_learning_parameters(
        episodes, alpha, gamma, epsilon, epsilon_decay, min_epsilon, max_steps
    )
    rng = np.random.default_rng(seed)
    q_values = np.zeros((env.observation_space.n, env.action_space.n), dtype=float)
    returns, lengths = [], []
    current_epsilon = epsilon

    for episode in range(episodes):
        state, _ = env.reset(seed=seed if episode == 0 else None)
        action = _epsilon_greedy(q_values, state, current_epsilon, rng)
        total_reward = 0.0
        for step in range(1, max_steps + 1):
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated or step == max_steps
            if done:
                target = reward
            else:
                next_action = _epsilon_greedy(
                    q_values, next_state, current_epsilon, rng
                )
                target = reward + gamma * q_values[next_state, next_action]
            q_values[state, action] += alpha * (target - q_values[state, action])
            total_reward += reward
            if done:
                break
            state, action = next_state, next_action
        returns.append(total_reward)
        lengths.append(step)
        current_epsilon = max(min_epsilon, current_epsilon * epsilon_decay)

    return _learning_result(q_values, returns, lengths, episodes)
