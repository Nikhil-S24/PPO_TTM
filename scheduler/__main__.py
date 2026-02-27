"""Built-in Fleet Scheduling Policies.
These classes can be extended for future research.
"""

import argparse
import yaml
import torch

from stable_baselines3 import PPO

from simulator.simulator import TaxiFleetSimulator
from scheduler.policies import (
    DataLogger,
    EightyTwentyPolicy,
    TTMEnhancedPolicy,
    DnnPolicy,
)

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
        help="Path to state output log (required for EVAL)"
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
        help="Number of training timesteps (TRAIN only)"
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

        env = TaxiFleetSimulator(config)
        env.reset()

        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=args.epochs)

        torch.save(model.policy, "ppo_ttm.pt")
        print("✅ PPO training complete. Model saved as ppo_ttm.pt")

    # ==================================================
    # EVAL MODE
    # ==================================================
    elif args.action.lower() == "eval":
        if args.output is None or args.policy is None:
            raise ValueError("--output and --policy must be specified for EVAL")

        datalogger = DataLogger(args.output)

        if args.policy.lower() == "eightytwenty":
            policy = EightyTwentyPolicy()

        elif args.policy.lower() == "ttm":
            policy = TTMEnhancedPolicy()

        elif args.policy.lower() == "dnn":
            if args.weights is None:
                raise ValueError("--weights required for DNN policy")
            policy = DnnPolicy(args.weights)

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
        print(f"✅ Evaluation complete. Output saved to {args.output}")

    else:
        raise ValueError("Action must be TRAIN or EVAL")
