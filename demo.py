"""Train or plan a policy, then run reproducible Icy Grid World episodes."""

import argparse
import time

import gymnasium as gym
import icy_gridworld  # Registers the environment.
from icy_gridworld.algorithms import policy_iteration, q_learning, sarsa, value_iteration


def show_animated_frame(env, episode, step, delay):
    """Refresh one terminal grid and pause so movement is visible."""
    print("\033[2J\033[H", end="")
    print(env.render())
    print(f"\nEpisode {episode} | step {step}")
    time.sleep(delay)


def make_policy(env, name, training_episodes, seed, alpha):
    """Create the requested policy and return it with optional training results."""
    if name == "random":
        return None, None
    if name in {"optimal", "value-iteration"}:
        result = value_iteration(env.unwrapped)
    elif name == "policy-iteration":
        result = policy_iteration(env.unwrapped)
    elif name == "sarsa":
        result = sarsa(env, episodes=training_episodes, alpha=alpha, seed=seed)
    else:
        result = q_learning(env, episodes=training_episodes, alpha=alpha, seed=seed)
    return result.policy, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--policy",
        choices=(
            "random", "optimal", "value-iteration", "policy-iteration", "sarsa", "q-learning"
        ),
        default="value-iteration",
    )
    parser.add_argument(
        "--training-episodes", type=int, default=3_000,
        help="training episodes for SARSA and Q-learning",
    )
    parser.add_argument("--alpha", type=float, default=0.1, help="learning rate")
    parser.add_argument("--render", choices=("none", "ansi", "human"), default="ansi")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="seconds between refreshed frames when --render human is used",
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.delay < 0:
        parser.error("--delay must be nonnegative")
    if args.delay and args.render != "human":
        parser.error("--delay requires --render human")
    if args.training_episodes < 1:
        parser.error("--training-episodes must be positive")
    if not 0 < args.alpha <= 1:
        parser.error("--alpha must be in (0, 1]")

    animated = args.render == "human" and args.delay > 0
    # ANSI returns frames without printing them, letting this demo refresh one
    # terminal grid instead of producing a fast scrolling stream.
    mode = "ansi" if animated else (None if args.render == "none" else args.render)
    env = gym.make("IcyGridWorld-v0", render_mode=mode)
    try:
        policy, result = make_policy(
            env, args.policy, args.training_episodes, args.seed, args.alpha
        )
        if result is not None and result.episode_returns:
            window = min(100, len(result.episode_returns))
            mean_return = sum(result.episode_returns[-window:]) / window
            print(
                f"Trained {args.policy} for {result.iterations} episodes; "
                f"mean return over final {window}: {mean_return:.2f}"
            )
        env.action_space.seed(args.seed)
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed if episode == 0 else None)
            total_reward = 0.0
            if animated:
                show_animated_frame(env, episode + 1, 0, args.delay)
            while True:
                action = env.action_space.sample() if policy is None else int(policy[observation])
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if animated:
                    show_animated_frame(env, episode + 1, info["steps"], args.delay)
                if terminated or truncated:
                    break
            if args.render == "ansi":
                print(env.render())
            print(f"Episode {episode + 1}: steps={info['steps']}, reward={total_reward:.1f}, "
                  f"terminated={terminated}, truncated={truncated}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
