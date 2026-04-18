"""Built-in Fleet Scheduling Policies.
These classes can be extended for future research.
"""

import argparse
import yaml
import gymnasium as gym

from stable_baselines3 import PPO

from simulator.simulator import TaxiFleetSimulator
from scheduler.policies import (
    DataLogger,
    EightyTwentyPolicy,
    TTMEnhancedPolicy,
    DnnPolicy,
)


class PPORewardWrapper(gym.Wrapper):
    """PPO-only reward shaping; does not affect baseline/TTM eval."""

    def __init__(self, env):
        super().__init__(env)
        self.prev_completed = 0
        self.prev_revenue = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_completed = info.get("completed", 0)
        self.prev_revenue = info.get("total_revenue", 0.0)
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        completed = info.get("completed", 0)
        revenue = info.get("total_revenue", 0.0)

        inc_completed = completed - self.prev_completed
        inc_revenue = revenue - self.prev_revenue

        # Revenue-focused dense reward for PPO
        reward = inc_revenue

        self.prev_completed = completed
        self.prev_revenue = revenue

        return obs, reward, terminated, truncated, info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate vehicle fleet")
    parser.add_argument(
        "-c", "--config", required=True,
        help="Path to configuration file for a simulation"
    )
    parser.add_argument(
        "-a", "--action", required=True,
        help="TRAIN or EVAL"
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to state output log (required for EVAL / TRAIN model path)"
    )
    parser.add_argument(
        "-p", "--policy",
        help="EIGHTYTWENTY, TTM or DNN (required for EVAL)"
    )
    parser.add_argument(
        "-w", "--weights",
        help="Path to policy weights for DNN / PPO"
    )
    parser.add_argument(
        "--epochs", type=int,
        help="Number of training epochs (TRAIN only)"
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Load configuration
    # --------------------------------------------------
    with open(args.config, "r") as fp:
        config = yaml.safe_load(fp)

    # ==================================================
    # TRAIN MODE (PPO)
    # ==================================================
    if args.action.lower() == "train":
        if args.epochs is None:
            raise ValueError("--epochs must be specified for TRAIN")

        # PPO-only reward shaping wrapper (does NOT affect baseline/TTM eval path)
        env = PPORewardWrapper(TaxiFleetSimulator(config))
        env.reset()

        # Scale training steps based on epochs
        total_steps = args.epochs * 10000
        print(f"Training PPO for {total_steps} timesteps...")

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            n_steps=4096,
            batch_size=1024,
            learning_rate=5e-5,
            gamma=0.99,
            ent_coef=0.05,
            clip_range=0.05,
        )

        model.learn(total_timesteps=total_steps)

        save_path = args.output if args.output else "ppo_model.pt"
        model.save(save_path)

        print(f"PPO training complete. Model saved as {save_path}")

    # ==================================================
    # EVAL MODE
    # ==================================================
    elif args.action.lower() == "eval":
        if args.output is None or args.policy is None:
            raise ValueError("--output and --policy must be specified for EVAL")

        datalogger = DataLogger(args.output, config["fleet"]["size"])

        if args.policy.lower() == "eightytwenty":
            policy = EightyTwentyPolicy()

        elif args.policy.lower() == "ttm":
            policy = TTMEnhancedPolicy()

        elif args.policy.lower() == "dnn":
            if args.weights is None:
                raise ValueError("--weights required for DNN policy")

            # ✅ Load PPO model directly
            model = PPO.load(args.weights)

            class PPOPolicyWrapper:
                def __init__(self, model):
                    self.model = model

                def schedule(self, observation, info):
                    action, _ = self.model.predict(observation, deterministic=True)
                    return action

            policy = PPOPolicyWrapper(model)

        else:
            raise Exception("Choose a supported policy!")

        environment = TaxiFleetSimulator(config)
        observation, info = environment.reset()
        done = False

        while not done:
            datalogger.write(info)
            action = policy.schedule(observation, info)
            observation, reward, done, _, info = environment.step(action)

        datalogger.close()
        print(f"Evaluation complete. Output saved to {args.output}")

    else:
        raise ValueError("Action must be TRAIN or EVAL")