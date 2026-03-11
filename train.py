import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np
import env # registers the environment
import optuna
from optuna.trial import TrialState

import pufferlib
import pufferlib.vector
import pufferlib.emulation
import pufferlib.models

from torch.utils.tensorboard import SummaryWriter

class Agent(nn.Module):
    def __init__(self, envs, arch="cnn"):
        super().__init__()
        self.arch = arch
        if arch == "cnn":
            self.conv_net = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Flatten()
            )
            self.fc = nn.Sequential(
                nn.Linear(4096 + 75, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU()
            )
        else: # MLP
            self.fc = nn.Sequential(
                nn.Linear(139, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU()
            )
        
        self.actor = nn.Linear(256, 192)
        self.critic = nn.Linear(256, 1)

    def _get_hidden(self, x):
        if self.arch == "cnn":
            board = x[:, :64].view(-1, 1, 8, 8).float()
            shapes = x[:, 64:].float()
            board_features = self.conv_net(board)
            return self.fc(torch.cat([board_features, shapes], dim=1))
        else:
            return self.fc(x.float())

    def get_value(self, x):
        return self.critic(self._get_hidden(x))

    def get_action_and_value(self, x, action=None, mask=None):
        hidden = self._get_hidden(x)
        logits = self.actor(hidden)
        if mask is not None:
            logits = logits + (mask == 0) * -1e9
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

def make_env(**kwargs):
    return pufferlib.emulation.GymnasiumPufferEnv(
        env_creator=lambda: env.BlockBlastEnv(render_mode=None),
        **kwargs
    )

def objective(trial):
    # Hyperparameters to tune
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.01, 0.1, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    arch = trial.suggest_categorical("arch", ["cnn", "mlp"])
    
    num_envs = 8
    num_steps = 128
    total_timesteps = 200000 # 200k steps for fast initial screening
    batch_size = int(num_envs * num_steps)
    num_iterations = total_timesteps // batch_size
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    envs = pufferlib.vector.make(
        make_env,
        num_envs=num_envs,
        backend=pufferlib.vector.Multiprocessing,
    )
    
    agent = Agent(envs, arch=arch).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
    
    # Create a descriptive run name
    run_name = f"Trial_{trial.number}_{arch}_LR{lr:.2e}_Ent{ent_coef:.2f}"
    log_dir = f"runs/{run_name}"
    writer = SummaryWriter(log_dir)
    
    # Professional HParams Logging (Prevents Duplicates/Nesting)
    from torch.utils.tensorboard.summary import hparams as hp_summary
    hparams = {"lr": lr, "ent_coef": ent_coef, "gamma": gamma, "arch": arch}
    metric_dict = {"charts/avg_reward": 0}
    exp, ssi, sei = hp_summary(hparams, metric_dict)
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)
    
    obs_size, mask_size = 139, 192
    obs_shape = envs.single_observation_space.shape
    
    # Storage for optimization (simplified PPO)
    obs_buffer = torch.zeros((num_steps, num_envs) + obs_shape).to(device)
    actions_buffer = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
    rewards_buffer = torch.zeros((num_steps, num_envs)).to(device)
    
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    
    final_reward = 0
    for iteration in range(1, num_iterations + 1):
        for step in range(0, num_steps):
            obs_buffer[step] = next_obs
            current_obs = next_obs[:, :obs_size]
            current_mask = next_obs[:, -mask_size:]

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(current_obs, mask=current_mask)
            
            actions_buffer[step] = action
            next_obs, reward, terminated, truncated, _ = envs.step(action.cpu().numpy())
            next_obs = torch.Tensor(next_obs).to(device)
            rewards_buffer[step] = torch.tensor(reward).to(device)

        # Optimization Step (PPO-style)
        b_obs = obs_buffer.reshape((-1,) + obs_shape)
        b_actions = actions_buffer.reshape((-1,) + envs.single_action_space.shape)
        b_returns = rewards_buffer.reshape(-1)
        
        b_current_obs = b_obs[:, :obs_size]
        b_current_mask = b_obs[:, -mask_size:]
        
        _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_current_obs, b_actions.long(), mask=b_current_mask)
        
        pg_loss = -newlogprob.mean()
        v_loss = ((newvalue.view(-1) - b_returns) ** 2).mean() * 0.5
        ent_loss = entropy.mean()
        loss = pg_loss + v_loss - ent_coef * ent_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        avg_reward = rewards_buffer.mean().item()
        trial.report(avg_reward, iteration)
        
        # Log every iteration for real-time live charting
        writer.add_scalar("charts/avg_reward", avg_reward, iteration)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), iteration)
        writer.add_scalar("losses/value_loss", v_loss.item(), iteration)
        writer.add_scalar("losses/entropy", ent_loss.item(), iteration)
        
        # Save Checkpoint in a dedicated run folder
        if iteration % 100 == 0 or iteration == num_iterations:
            checkpoint_dir = f"checkpoints/{run_name}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = f"{checkpoint_dir}/iter_{iteration}.pt"
            torch.save(agent.state_dict(), checkpoint_path)
            
        # Pruning
        if trial.should_prune():
            envs.close()
            writer.close()
            raise optuna.exceptions.TrialPruned()
            
        final_reward = avg_reward
        if iteration % 50 == 0:
            print(f"Trial {trial.number} | Iter {iteration} | Avg Reward: {avg_reward:.2f}")

    envs.close()
    writer.close()
    return final_reward

def main():
    # 1. Create Study with Persistence (SQLite)
    db_path = "sqlite:///optuna_study.db"
    study = optuna.create_study(
        study_name="blockblast_sweep",
        storage=db_path,
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=100)
    )
    
    # 2. Run Optimization
    print("Starting Optuna Hyperparameter Sweep...")
    study.optimize(objective, n_trials=10) # Test 10 different combinations
    
    # 3. Print Results
    print("\nSweep Complete!")
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Reward: {study.best_value:.2f}")
    print(f"Best Params: {study.best_params}")
    
    # 4. Save Visualization
    try:
        import optuna.visualization as vis
        fig = vis.plot_contour(study, params=["lr", "ent_coef"])
        fig.write_html("optuna_surface_plot.html")
        print("Surface plot saved to: optuna_surface_plot.html")
    except Exception as e:
        print(f"Could not save plot: {e}")

if __name__ == "__main__":
    main()