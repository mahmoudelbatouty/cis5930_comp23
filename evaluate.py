"""Compare planning, model-free learning, and random policies."""

import argparse
import statistics

import gymnasium as gym

import icy_gridworld  # Registers the environment.
from icy_gridworld.algorithms import policy_iteration, q_learning, sarsa, value_iteration
from icy_gridworld.dqn import dqn


def train_policy(
    policy_name, training_episodes, training_seed, dqn_training_episodes=1_000
):
    if policy_name == "random":
        return None
    env = gym.make("IcyGridWorld-v0")
    try:
        if policy_name == "value-iteration":
            return value_iteration(env.unwrapped).policy
        if policy_name == "policy-iteration":
            return policy_iteration(env.unwrapped).policy
        if policy_name == "sarsa":
            return sarsa(env, episodes=training_episodes, seed=training_seed).policy
        if policy_name == "dqn":
            return dqn(env, episodes=dqn_training_episodes, seed=training_seed).policy
        return q_learning(env, episodes=training_episodes, seed=training_seed).policy
    finally:
        env.close()


def evaluate(
    policy_name,
    seeds,
    training_episodes=3_000,
    training_seed=42,
    dqn_training_episodes=1_000,
):
    env = gym.make("IcyGridWorld-v0")
    policy = train_policy(
        policy_name, training_episodes, training_seed, dqn_training_episodes
    )
    results = []
    try:
        for seed in range(seeds):
            observation, _ = env.reset(seed=seed)
            env.action_space.seed(seed)
            total_reward = 0.0
            while True:
                action = env.action_space.sample() if policy is None else int(policy[observation])
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    results.append((info["steps"], total_reward, terminated, truncated))
                    break
    finally:
        env.close()

    successes = sum(result[2] for result in results)
    truncations = sum(result[3] for result in results)
    return {
        "successes": successes,
        "mean_steps": statistics.mean(result[0] for result in results),
        "mean_return": statistics.mean(result[1] for result in results),
        "truncations": truncations,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--training-episodes", type=int, default=3_000)
    parser.add_argument("--dqn-training-episodes", type=int, default=1_000)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=("value-iteration", "policy-iteration", "sarsa", "q-learning", "dqn", "random"),
        default=("value-iteration", "policy-iteration", "sarsa", "q-learning", "dqn", "random"),
    )
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    if args.training_episodes < 1:
        parser.error("--training-episodes must be positive")
    if args.dqn_training_episodes < 1:
        parser.error("--dqn-training-episodes must be positive")

    for policy_name in args.algorithms:
        result = evaluate(
            policy_name,
            args.seeds,
            args.training_episodes,
            args.training_seed,
            args.dqn_training_episodes,
        )
        print(
            f"{policy_name}: success={result['successes']}/{args.seeds}, "
            f"mean_steps={result['mean_steps']:.2f}, "
            f"mean_return={result['mean_return']:.2f}, "
            f"truncated={result['truncations']}/{args.seeds}"
        )


if __name__ == "__main__":
    main()
