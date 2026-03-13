import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np
import env
import pufferlib
import pufferlib.vector
import pufferlib.emulation
import optuna
from torch.utils.tensorboard import SummaryWriter

class FlexibleAgent(nn.Module):
    def __init__(self, obs_size=139, action_size=192, 
                 arch_type="cnn", 
                 cnn_channels=[32, 64], 
                 fc_layers=[512, 512, 256],
                 lstm_hidden=0,
                 transformer_layers=0,
                 transformer_heads=4,
                 activation="relu"):
        super().__init__()
        self.arch_type = arch_type
        self.lstm_hidden = lstm_hidden
        self.transformer_layers = transformer_layers
        self.obs_size = obs_size
        
        act_fn = nn.ReLU if activation == "relu" else nn.GELU
        
        if arch_type == "cnn":
            self.local_conv = nn.Sequential(
                nn.Conv2d(1, cnn_channels[0], kernel_size=3, padding=1),
                act_fn(),
                nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1),
                act_fn(),
                nn.Flatten()
            )
            self.global_conv = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=8),
                act_fn(),
                nn.Flatten()
            )
            self.row_conv = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(1, 8)),
                act_fn(),
                nn.Flatten()
            )
            self.col_conv = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(8, 1)),
                act_fn(),
                nn.Flatten()
            )
            self.input_dim = cnn_channels[1] * 64 + 32 + 128 + 128 + 75
        else:
            self.input_dim = obs_size
        
        layers = []
        last_dim = self.input_dim
        for hidden in fc_layers:
            layers.append(nn.Linear(last_dim, hidden))
            layers.append(act_fn())
            last_dim = hidden
        
        self.fc = nn.Sequential(*layers)
        
        if transformer_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=last_dim, 
                nhead=transformer_heads, 
                dim_feedforward=last_dim*2,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        if lstm_hidden > 0:
            self.lstm = nn.LSTM(last_dim, lstm_hidden)
            last_dim = lstm_hidden
            
        self.actor = nn.Linear(last_dim, action_size)
        self.critic = nn.Linear(last_dim, 1)

    def _get_hidden(self, x, lstm_state=None):
        if self.arch_type == "cnn":
            board = x[:, :64].view(-1, 1, 8, 8).float()
            shapes = x[:, 64:139].float()
            l_feat = self.local_conv(board)
            g_feat = self.global_conv(board)
            r_feat = self.row_conv(board)
            c_feat = self.col_conv(board)
            features = torch.cat([l_feat, g_feat, r_feat, c_feat, shapes], dim=1)
        else:
            features = x.float()
            
        hidden = self.fc(features)

        if self.transformer_layers > 0:
            hidden = hidden.unsqueeze(1) # (B, 1, D)
            hidden = self.transformer(hidden)
            hidden = hidden.squeeze(1)
        
        if self.lstm_hidden > 0:
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)
                hidden, lstm_state = self.lstm(hidden, lstm_state)
                hidden = hidden.squeeze(0)
            else:
                hidden, lstm_state = self.lstm(hidden, lstm_state)
        
        return hidden, lstm_state

    def get_value(self, x, lstm_state=None):
        hidden, _ = self._get_hidden(x, lstm_state)
        return self.critic(hidden)

    def get_action_and_value(self, x, action=None, mask=None, lstm_state=None):
        hidden, next_lstm_state = self._get_hidden(x, lstm_state)
        logits = self.actor(hidden)
        if mask is not None:
            logits = logits + (mask == 0) * -1e9
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), next_lstm_state

def make_env(**kwargs):
    return pufferlib.emulation.GymnasiumPufferEnv(
        env_creator=lambda: env.BlockBlastEnv(render_mode=None),
        **kwargs
    )

def train(trial):
    # Hyperparameters to tune
    lr = trial.suggest_float("lr", 5e-5, 5e-4, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.001, 0.05, log=True)
    gamma = trial.suggest_float("gamma", 0.99, 0.999)
    activation = trial.suggest_categorical("activation", ["relu", "gelu"])
    arch_type = trial.suggest_categorical("arch_type", ["cnn", "mlp"])
    
    num_fc = trial.suggest_int("num_fc", 1, 3)
    fc_dim = trial.suggest_categorical("fc_dim", [256, 512])
    fc_layers = [fc_dim] * num_fc
    
    use_lstm = trial.suggest_categorical("use_lstm", [True, False])
    lstm_hidden = trial.suggest_categorical("lstm_hidden", [64, 128]) if use_lstm else 0
    
    use_transformer = trial.suggest_categorical("use_transformer", [True, False])
    transformer_layers = trial.suggest_int("transformer_layers", 1, 2) if use_transformer else 0
    transformer_heads = trial.suggest_categorical("transformer_heads", [4, 8]) if use_transformer else 4
    
    cnn_channels = [32, 64]
    if arch_type == "cnn":
        cnn_channels = [
            trial.suggest_categorical("cnn_c1", [16, 32]),
            trial.suggest_categorical("cnn_c2", [32, 64])
        ]

    num_envs = 128
    num_steps = 256
    total_updates = 100 # ~3.3 million steps (128 * 256 * 100 = 3,276,800)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Unique run name for TensorBoard
    fc_str = "x".join(map(str, fc_layers))
    cnn_str = f"C{cnn_channels[0]}x{cnn_channels[1]}" if arch_type == "cnn" else ""
    lstm_str = f"_LSTM{lstm_hidden}" if lstm_hidden > 0 else ""
    trans_str = f"_T{transformer_layers}" if transformer_layers > 0 else ""
    trial_run_name = f"Trial{trial.number}_{arch_type.upper()}_{cnn_str}_FC{fc_str}{lstm_str}{trans_str}_{activation.upper()}_LR{lr:.5f}"
    writer = SummaryWriter(f"runs/optuna/{trial_run_name}")

    envs = pufferlib.vector.make(make_env, num_envs=num_envs, backend=pufferlib.vector.Serial)
    agent = FlexibleAgent(
        arch_type=arch_type, 
        cnn_channels=cnn_channels, 
        fc_layers=fc_layers, 
        lstm_hidden=lstm_hidden,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        activation=activation
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
    
    obs_size, mask_size = 139, 192
    obs_shape = envs.single_observation_space.shape
    
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(num_envs).to(device)
    
    gae_lambda = 0.95
    num_minibatches = 4
    update_epochs = 4
    clip_coef = 0.2
    vf_coef = 0.5
    max_grad_norm = 0.5
    
    batch_size = int(num_envs * num_steps)
    minibatch_size = int(batch_size // num_minibatches)
    
    recent_rewards = []
    start_time = time.time()

    for update in range(1, total_updates + 1):
        obs_buffer = torch.zeros((num_steps, num_envs) + obs_shape).to(device)
        actions_buffer = torch.zeros((num_steps, num_envs)).to(device)
        logprobs_buffer = torch.zeros((num_steps, num_envs)).to(device)
        rewards_buffer = torch.zeros((num_steps, num_envs)).to(device)
        dones_buffer = torch.zeros((num_steps, num_envs)).to(device)
        values_buffer = torch.zeros((num_steps, num_envs)).to(device)

        for step in range(0, num_steps):
            obs_buffer[step] = next_obs
            dones_buffer[step] = next_done
            current_mask = next_obs[:, :mask_size]
            current_obs = next_obs[:, mask_size : mask_size + obs_size]

            with torch.no_grad():
                action, logprob, _, value, _ = agent.get_action_and_value(current_obs, mask=current_mask)
                values_buffer[step] = value.flatten()
            
            actions_buffer[step] = action
            logprobs_buffer[step] = logprob
        next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
        next_done = torch.logical_or(torch.Tensor(terminated), torch.Tensor(truncated)).to(device)
        next_obs = torch.Tensor(next_obs).to(device)
        rewards_buffer[step] = torch.tensor(reward).to(device)
        
        # Track lines cleared in Optuna for universal comparison
        if "lines_cleared" in infos:
            writer.add_scalar("charts/total_lines_cleared", np.mean(infos["lines_cleared"]), global_step)

        with torch.no_grad():
            next_obs_actual = next_obs[:, mask_size : mask_size + obs_size]
            next_value = agent.get_value(next_obs_actual).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buffer).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done.float()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buffer[t + 1]
                    nextvalues = values_buffer[t + 1]
                delta = rewards_buffer[t] + gamma * nextvalues * nextnonterminal - values_buffer[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_buffer

        b_obs = obs_buffer.reshape((-1,) + obs_shape)
        b_logprobs = logprobs_buffer.reshape(-1)
        b_actions = actions_buffer.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_current_obs = b_obs[:, mask_size : mask_size + obs_size]
        b_current_mask = b_obs[:, :mask_size]

        b_inds = np.arange(batch_size)
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_current_obs[mb_inds], 
                    b_actions.long()[mb_inds], 
                    mask=b_current_mask[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()
                loss = pg_loss - ent_coef * entropy.mean() + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

        avg_reward = rewards_buffer.mean().item()
        recent_rewards.append(avg_reward)
        if len(recent_rewards) > 10:
            recent_rewards.pop(0)
            
        global_step = update * batch_size
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/avg_reward", avg_reward, global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy.mean().item(), global_step)

        trial.report(np.mean(recent_rewards), update)
        if trial.should_prune():
            envs.close()
            writer.close()
            raise optuna.exceptions.TrialPruned()

    envs.close()
    writer.close()
    
    # Save the final trial model so it can be played/tested
    checkpoint_dir = f"checkpoints/optuna/{trial_run_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save({
        "agent_state_dict": agent.state_dict(),
        "update": total_updates,
        "global_step": total_updates * batch_size,
    }, os.path.join(checkpoint_dir, "model.pt"))
    
    # Save config for play.py
    import json
    config = {
        "arch": arch_type,
        "fc_layers": fc_layers,
        "cnn_channels": cnn_channels,
        "lstm": lstm_hidden,
        "transformer_layers": transformer_layers,
        "transformer_heads": transformer_heads,
        "activation": activation,
        "ent_coef": ent_coef,
        "gamma": gamma,
        "lr": lr
    }
    with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    return np.mean(recent_rewards)

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(train, n_trials=10)
    
    print("Best Params:", study.best_params)
    print("Best Value:", study.best_value)

    print("\n" + "="*50)
    print("STARTING FULL TRAINING WITH BEST PARAMETERS")
    print("="*50)
    
    import subprocess
    best = study.best_params
    
        cmd = [
    
            "python", "train.py",
    
            "--arch", str(best["arch_type"]),
    
            "--lr", str(best["lr"]),
    
            "--ent-coef", str(best["ent_coef"]),
    
            "--gamma", str(best["gamma"]),
    
            "--activation", str(best["activation"]),
    
            "--fc-layers"
    
        ]
    
     + [str(best["fc_dim"])] * best["num_fc"]
    
    if best["arch_type"] == "cnn":
        cmd += ["--cnn-channels", str(best["cnn_c1"]), str(best["cnn_c2"])]
    
    if best["use_lstm"]:
        cmd += ["--lstm", str(best["lstm_hidden"])]
        
    if best["use_transformer"]:
        cmd += ["--transformer-layers", str(best["transformer_layers"]), "--transformer-heads", str(best["transformer_heads"])]
        
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)