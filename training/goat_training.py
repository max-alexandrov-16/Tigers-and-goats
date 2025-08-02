# training/goat_training.py

import os
import json
import torch
import numpy as np
import traceback
from datetime import datetime
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from argparse import Namespace
from agents.goat_agent import ActorCriticGoatAgent
from models.goat_model import ActorModel,CNNActorModel, CriticModel
from helpers.goat_env import GOAT_ENV


def save_metrics(log_path, run_index, avg_reward, critic_loss):
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "run": run_index,
            "avg_reward": avg_reward,
            "critic_loss": critic_loss
        }) + "\n")



args = Namespace(
    runs=200000,
    log_step=2000,
    max_turns=30,
    size=5
)


def run_single_experiment(actor_lr, critic_lr, critic_weight_decay, args, tag_suffix=""):
    torch.manual_seed(2)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"critic_lr_{critic_lr}_actor_lr_{actor_lr}_critic_wd_{critic_weight_decay}_{tag_suffix}_{timestamp}"
    base_dir = os.path.join("experiments", experiment_name)
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    config_path = os.path.join(base_dir, "config.json")
    config_dict = {
        "actor_lr": actor_lr,
        "critic_lr": critic_lr,
        "critic_weight_decay": critic_weight_decay,
        "max_turns": args.max_turns,
        "size": args.size,
        "runs": args.runs,
        "log_step": args.log_step
    }
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=4)

    log_file = os.path.join(log_dir, "metrics.jsonl")

    goat_reward_scheme = {"winning": 1, "losing": -1}
    tiger_reward_scheme = {"eating": 0.1, "winning": 0.5, "losing": -0.5, "no score": 0}
    goat_env = GOAT_ENV(args.size, args.max_turns, goat_reward_scheme, tiger_reward_scheme)

    actor_model = CNNActorModel(args.size)
    critic_model = CriticModel(args.size)
    actor_optimizer = torch.optim.Adam(actor_model.parameters(), lr=actor_lr)
    critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr=critic_lr, weight_decay=critic_weight_decay)

    goat_agent = ActorCriticGoatAgent(
        actor_model, critic_model,
        goat_env, args.size,
        goat_reward_scheme,
        actor_optimizer, critic_optimizer,
        args.max_turns
    )

    avg_rewards = []
    critic_losses = []

    for i in range(args.runs):
        try:
            avg_reward, critic_loss = goat_agent.learn()
            goat_agent.clear_memory()
            avg_rewards.append(avg_reward)
            critic_losses.append(critic_loss)

            if i % args.log_step == 0 and i != 0:
                avg_r = np.mean(avg_rewards[-args.log_step:])
                avg_l = np.mean(critic_losses[-args.log_step:])
                print(f"[{experiment_name} | Run {i}] Avg Reward: {avg_r:.4f} | Critic Loss: {avg_l:.4f}")
                save_metrics(log_file, i, avg_r, avg_l)

                # === Save checkpoint ===
                checkpoint = {
                    'actor_model_state_dict': actor_model.state_dict(),
                    'critic_model_state_dict': critic_model.state_dict(),
                    'actor_optimizer_state_dict': actor_optimizer.state_dict(),
                    'critic_optimizer_state_dict': critic_optimizer.state_dict(),
                    'step': i
                }

                torch.save(checkpoint, os.path.join(checkpoint_dir, f"checkpoint_step_{i}.pt"))

                # === Save raw weights separately (optional, for inference/loading convenience) ===
                torch.save(actor_model.state_dict(), os.path.join(checkpoint_dir, f"actor_weights_step_{i}.pth"))
                torch.save(critic_model.state_dict(), os.path.join(checkpoint_dir, f"critic_weights_step_{i}.pth"))


        except Exception as e:
            print(f"[{experiment_name} | Run {i}] Exception: {e}")
            traceback.print_exc()

            # === Save in-memory episode traces ===
            states, actions, rewards, log_probs, values = goat_agent.get_memory()
            memory_trace = {
                "states": states,
                "actions": actions,
                "rewards": rewards,
                "log_probs": log_probs,
                "values": values
            }
            memory_path = os.path.join(log_dir, f"episode_memory_step_{i}.pt")
            torch.save(memory_trace, memory_path)

            break

HYPERPARAMETER_TUPLES = [
    (1e-5, 1e-4, 1e-5), # actor_lr, critic_lr, critic_weight_decay
    (1e-4, 1e-3, 1e-4),
    (1e-4, 1e-3, 1e-5)
]

def main(args):

    for i, (actor_lr, critic_lr, weight_decay) in enumerate(HYPERPARAMETER_TUPLES):
        run_single_experiment(
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            critic_weight_decay=weight_decay,
            args=args,
            tag_suffix=f"trial_{i}"
        )


if __name__ == "__main__":
    main(args)
