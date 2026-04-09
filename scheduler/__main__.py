"""Built-in Fleet Scheduling Policies.
These classes can be extended for future research.
"""

import argparse
import yaml
import torch
import numpy as np
import gymnasium as gym # Added to define the new action space

from stable_baselines3 import PPO
from gymnasium.wrappers import TransformAction

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

        env = TaxiFleetSimulator(config)
        
        # ✅ CORRECTED STEP 2: SCALE TRAINING ACTIONS WITH ACTION_SPACE
        max_power = config["charging stations"][0]["max port power"] 
        fleet_size = config["fleet"]["size"]
        
        # Define the new action space bounds for the wrapper
        # The PPO model will output values in [0, 1], but we transform them
        # so the second column (power) reaches max_power.
        low = np.zeros((fleet_size, 2), dtype=np.float32)
        high = np.ones((fleet_size, 2), dtype=np.float32)
        high[:, 1] = max_power
        new_action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        def scale_action(act):
            # act is a (fleet_size, 2) array.
            scaled = act.copy()
            # Scaling only the second column (power) by max_power
            scaled[:, 1] *= max_power
            return scaled
            
        # The wrapper now correctly receives the new_action_space
        env = TransformAction(env, scale_action, new_action_space)
        env.reset()

        # Scale training steps based on epochs
        total_steps = args.epochs * 1000
        print(f"🚀 Training PPO for {total_steps} timesteps (Max Power: {max_power}kW)...")

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            n_steps=2048,
            batch_size=256,
            learning_rate=3e-4,
            gamma=0.99,
        )

        model.learn(total_timesteps=total_steps)

        # ✅ Save properly using output argument
        save_path = args.output if args.output else "ppo_model.pt"
        torch.save(model.policy, save_path)

        print(f"✅ PPO training complete. Model saved as {save_path}")

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