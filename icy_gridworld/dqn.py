"""A small NumPy implementation of Deep Q-Network control."""

import numpy as np

from icy_gridworld.algorithms import AlgorithmResult


def _encode_states(states):
    """Encode row and column as two concatenated one-hot vectors."""
    states = np.asarray(states, dtype=int).reshape(-1)
    features = np.zeros((len(states), 40), dtype=np.float32)
    rows, columns = np.divmod(states, 20)
    indices = np.arange(len(states))
    features[indices, rows] = 1.0
    features[indices, 20 + columns] = 1.0
    return features


class _Network:
    """Two-hidden-layer ReLU network with an Adam optimizer."""

    def __init__(self, action_count, hidden_size, learning_rate, rng):
        self.learning_rate = learning_rate
        self.parameters = {
            "w1": rng.normal(0, np.sqrt(2 / 40), (40, hidden_size)).astype(np.float32),
            "b1": np.zeros(hidden_size, dtype=np.float32),
            "w2": rng.normal(
                0, np.sqrt(2 / hidden_size), (hidden_size, hidden_size)
            ).astype(np.float32),
            "b2": np.zeros(hidden_size, dtype=np.float32),
            "w3": rng.normal(
                0, np.sqrt(2 / hidden_size), (hidden_size, action_count)
            ).astype(np.float32),
            "b3": np.zeros(action_count, dtype=np.float32),
        }
        self.first_moments = {
            name: np.zeros_like(value) for name, value in self.parameters.items()
        }
        self.second_moments = {
            name: np.zeros_like(value) for name, value in self.parameters.items()
        }
        self.optimizer_steps = 0

    def predict(self, states):
        x = _encode_states(states)
        hidden1 = np.maximum(x @ self.parameters["w1"] + self.parameters["b1"], 0)
        hidden2 = np.maximum(
            hidden1 @ self.parameters["w2"] + self.parameters["b2"], 0
        )
        return hidden2 @ self.parameters["w3"] + self.parameters["b3"]

    def train(self, states, actions, targets):
        x = _encode_states(states)
        z1 = x @ self.parameters["w1"] + self.parameters["b1"]
        hidden1 = np.maximum(z1, 0)
        z2 = hidden1 @ self.parameters["w2"] + self.parameters["b2"]
        hidden2 = np.maximum(z2, 0)
        predictions = hidden2 @ self.parameters["w3"] + self.parameters["b3"]

        batch_indices = np.arange(len(states))
        errors = predictions[batch_indices, actions] - targets
        output_gradient = np.zeros_like(predictions)
        output_gradient[batch_indices, actions] = np.clip(errors, -1.0, 1.0) / len(states)

        gradients = {}
        gradients["w3"] = hidden2.T @ output_gradient
        gradients["b3"] = output_gradient.sum(axis=0)
        hidden2_gradient = output_gradient @ self.parameters["w3"].T
        hidden2_gradient[z2 <= 0] = 0
        gradients["w2"] = hidden1.T @ hidden2_gradient
        gradients["b2"] = hidden2_gradient.sum(axis=0)
        hidden1_gradient = hidden2_gradient @ self.parameters["w2"].T
        hidden1_gradient[z1 <= 0] = 0
        gradients["w1"] = x.T @ hidden1_gradient
        gradients["b1"] = hidden1_gradient.sum(axis=0)

        self.optimizer_steps += 1
        beta1, beta2 = 0.9, 0.999
        for name, gradient in gradients.items():
            gradient = np.clip(gradient, -10.0, 10.0)
            self.first_moments[name] = (
                beta1 * self.first_moments[name] + (1 - beta1) * gradient
            )
            self.second_moments[name] = (
                beta2 * self.second_moments[name] + (1 - beta2) * gradient * gradient
            )
            corrected_first = self.first_moments[name] / (
                1 - beta1 ** self.optimizer_steps
            )
            corrected_second = self.second_moments[name] / (
                1 - beta2 ** self.optimizer_steps
            )
            self.parameters[name] -= self.learning_rate * corrected_first / (
                np.sqrt(corrected_second) + 1e-8
            )

    def copy_from(self, other):
        for name in self.parameters:
            self.parameters[name][...] = other.parameters[name]


class _ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.states = np.empty(capacity, dtype=np.int16)
        self.actions = np.empty(capacity, dtype=np.int8)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_states = np.empty(capacity, dtype=np.int16)
        self.done = np.empty(capacity, dtype=bool)
        self.size = 0
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        index = self.position
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.done[index] = done
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        indices = rng.choice(self.size, size=batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.done[indices],
        )


def _validate_parameters(
    episodes,
    learning_rate,
    gamma,
    epsilon,
    epsilon_decay,
    min_epsilon,
    max_steps,
    hidden_size,
    replay_capacity,
    batch_size,
    warmup_steps,
    train_frequency,
    target_update_frequency,
):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0 <= gamma < 1:
        raise ValueError("gamma must be in [0, 1)")
    if not 0 <= min_epsilon <= epsilon <= 1:
        raise ValueError("epsilon values must satisfy 0 <= min_epsilon <= epsilon <= 1")
    if not 0 < epsilon_decay <= 1:
        raise ValueError("epsilon_decay must be in (0, 1]")
    if max_steps < 1 or hidden_size < 1:
        raise ValueError("max_steps and hidden_size must be positive")
    if replay_capacity < 1 or batch_size < 1 or batch_size > replay_capacity:
        raise ValueError("replay capacity must be positive and at least batch_size")
    if warmup_steps < batch_size:
        raise ValueError("warmup_steps must be at least batch_size")
    if train_frequency < 1 or target_update_frequency < 1:
        raise ValueError("update frequencies must be positive")


def dqn(
    env,
    episodes=1_000,
    learning_rate=0.001,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.997,
    min_epsilon=0.05,
    seed=42,
    max_steps=500,
    hidden_size=64,
    replay_capacity=50_000,
    batch_size=64,
    warmup_steps=1_000,
    train_frequency=4,
    target_update_frequency=1_000,
    reward_scale=0.01,
):
    """Learn a greedy policy with experience replay and a target network.

    The network receives one-hot row and column features, and has two ReLU
    hidden layers. Rewards are scaled during optimization for numerical
    stability; returned Q-values are converted back to environment units.
    """
    _validate_parameters(
        episodes,
        learning_rate,
        gamma,
        epsilon,
        epsilon_decay,
        min_epsilon,
        max_steps,
        hidden_size,
        replay_capacity,
        batch_size,
        warmup_steps,
        train_frequency,
        target_update_frequency,
    )
    if reward_scale <= 0:
        raise ValueError("reward_scale must be positive")

    rng = np.random.default_rng(seed)
    online = _Network(env.action_space.n, hidden_size, learning_rate, rng)
    target = _Network(env.action_space.n, hidden_size, learning_rate, rng)
    target.copy_from(online)
    replay = _ReplayBuffer(replay_capacity)
    returns, lengths = [], []
    current_epsilon = epsilon
    total_steps = 0

    for episode in range(episodes):
        state, _ = env.reset(seed=seed if episode == 0 else None)
        total_reward = 0.0
        for step in range(1, max_steps + 1):
            if rng.random() < current_epsilon:
                action = int(rng.integers(env.action_space.n))
            else:
                action = int(online.predict([state])[0].argmax())

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated or step == max_steps
            replay.add(state, action, reward * reward_scale, next_state, done)
            total_reward += reward
            total_steps += 1

            if (
                replay.size >= warmup_steps
                and total_steps % train_frequency == 0
            ):
                states, actions, rewards, next_states, dones = replay.sample(
                    batch_size, rng
                )
                future = target.predict(next_states).max(axis=1)
                targets = rewards + gamma * future * (~dones)
                online.train(states, actions, targets)

            if total_steps % target_update_frequency == 0:
                target.copy_from(online)

            state = next_state
            if done:
                break

        returns.append(total_reward)
        lengths.append(step)
        current_epsilon = max(min_epsilon, current_epsilon * epsilon_decay)

    all_states = np.arange(env.observation_space.n)
    q_values = online.predict(all_states).astype(float) / reward_scale
    return AlgorithmResult(
        policy=q_values.argmax(axis=1),
        values=q_values.max(axis=1),
        q_values=q_values,
        iterations=episodes,
        episode_returns=tuple(returns),
        episode_lengths=tuple(lengths),
    )
