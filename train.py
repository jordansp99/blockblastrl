import os
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym")
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
import itertools
import subprocess
import webbrowser
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
            # We treat the hidden features as a sequence of tokens for the transformer
            # For simplicity, we use the last FC dimension as d_model
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
            # Transformer expects (Batch, Seq, Feature). 
            # We'll treat the hidden vector as a single token or split it?
            # Let's treat it as a sequence of 4 tokens if possible, or just 1 global token.
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a checkpoint to load")
    parser.add_argument("--run-name", type=str, default=None, help="The name of the run (for TensorBoard)")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000_000, help="Total timesteps")
    parser.add_argument("--no-tensorboard", action="store_true", help="Don't launch tensorboard")
    parser.add_argument("--start-update", type=int, default=None, help="Manually set the starting update number")
    
    # Architecture arguments
    parser.add_argument("--arch", type=str, default="cnn", choices=["cnn", "mlp"], help="Architecture type")
    parser.add_argument("--fc-layers", type=int, nargs="+", default=[256, 256, 256], help="FC layer dimensions")
    parser.add_argument("--cnn-channels", type=int, nargs=2, default=[16, 32], help="CNN channels")
    parser.add_argument("--lstm", type=int, default=0, help="LSTM hidden size (0 to disable)")
    parser.add_argument("--transformer-layers", type=int, default=0, help="Transformer encoder layers")
    parser.add_argument("--transformer-heads", type=int, default=4, help="Transformer attention heads")
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "gelu"], help="Activation function")
    parser.add_argument("--ent-coef", type=float, default=0.00895357273417024, help="Entropy coefficient for PPO")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma")
    parser.add_argument("--lr", type=float, default=0.0004981024654599184, help="Learning rate")

    args = parser.parse_args()

    # PPO CONFIGURATION
    num_envs = 128 
    num_steps = 256 
    total_timesteps = args.total_timesteps
    learning_rate = args.lr
    anneal_lr = True
    gamma = args.gamma
    gae_lambda = 0.95
    num_minibatches = 4
    update_epochs = 4
    norm_adv = True
    clip_coef = 0.2
    ent_coef = args.ent_coef
    vf_coef = 0.5
    max_grad_norm = 0.5
    
    batch_size = int(num_envs * num_steps)
    minibatch_size = int(batch_size // num_minibatches)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting PPO training on {device}...")
    
    envs = pufferlib.vector.make(make_env, num_envs=num_envs, backend=pufferlib.vector.Serial)
    agent = FlexibleAgent(
        arch_type=args.arch, 
        cnn_channels=args.cnn_channels, 
        fc_layers=args.fc_layers, 
        lstm_hidden=args.lstm,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        activation=args.activation
    ).to(device)
    
    # Generate run_name if not provided
    if not args.run_name:
        fc_str = "x".join(map(str, args.fc_layers))
        cnn_str = f"C{args.cnn_channels[0]}x{args.cnn_channels[1]}" if args.arch == "cnn" else ""
        lstm_str = f"_LSTM{args.lstm}" if args.lstm > 0 else ""
        trans_str = f"_T{args.transformer_layers}H{args.transformer_heads}" if args.transformer_layers > 0 else ""
        act_str = f"_{args.activation.upper()}"
        args.run_name = f"{args.arch.upper()}_{cnn_str}_FC{fc_str}{lstm_str}{trans_str}{act_str}_E{args.ent_coef}_LR{args.lr}_{int(time.time())}"
    
    run_name = args.run_name
    checkpoint_dir = f"checkpoints/{run_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save config for play.py to use automatically
    import json
    config = {
        "arch": args.arch,
        "fc_layers": args.fc_layers,
        "cnn_channels": args.cnn_channels,
        "lstm": args.lstm,
        "transformer_layers": args.transformer_layers,
        "transformer_heads": args.transformer_heads,
        "activation": args.activation,
        "ent_coef": args.ent_coef,
        "gamma": args.gamma,
        "lr": args.lr
    }
    with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
        
    writer = SummaryWriter(f"runs/{run_name}")
    
    if not args.no_tensorboard:
        print(f"Launching TensorBoard for {run_name}...")
        subprocess.Popen(["tensorboard", "--logdir=runs", "--port=6006"], 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2) # Wait for TB to start
        webbrowser.open("http://localhost:6006")
    
    obs_size, mask_size = 139, 192
    obs_shape = envs.single_observation_space.shape
    
    # Buffers
    obs_buffer = torch.zeros((num_steps, num_envs) + obs_shape).to(device)
    actions_buffer = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
    logprobs_buffer = torch.zeros((num_steps, num_envs)).to(device)
    rewards_buffer = torch.zeros((num_steps, num_envs)).to(device)
    dones_buffer = torch.zeros((num_steps, num_envs)).to(device)
    values_buffer = torch.zeros((num_steps, num_envs)).to(device)
    
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(num_envs).to(device)
    
    start_update = 1
    global_step = 0
    
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            checkpoint = torch.load(args.checkpoint, map_location=device)
            if isinstance(checkpoint, dict) and "agent_state_dict" in checkpoint:
                agent.load_state_dict(checkpoint["agent_state_dict"])
                start_update = checkpoint.get("update", 0) + 1
                global_step = checkpoint.get("global_step", 0)
                print(f"Loaded checkpoint: {args.checkpoint} (Resuming from update {start_update})")
            else:
                agent.load_state_dict(checkpoint)
                import re
                match = re.search(r"update_(\d+)", os.path.basename(args.checkpoint))
                if match:
                    start_update = int(match.group(1)) + 1
                    global_step = (start_update - 1) * batch_size
                    print(f"Loaded checkpoint: {args.checkpoint} (Legacy format, guessed update {start_update})")
                else:
                    print(f"Loaded checkpoint: {args.checkpoint} (Legacy format)")
        else:
            print(f"Error: Checkpoint {args.checkpoint} not found.")
            return

    if args.start_update is not None:
        start_update = args.start_update
        global_step = (start_update - 1) * batch_size
        print(f"Manually overriding starting update to {start_update}")

    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"Loaded optimizer state from {args.checkpoint}")

    total_episodes = 0
    episode_rewards = np.zeros(num_envs)
    episode_lengths = np.zeros(num_envs)
    
    # Track recent stats for console printing
    recent_returns = []
    recent_lengths = []
    recent_lines = []
    
    start_time = time.time()
    num_updates = total_timesteps // batch_size
    
    for update in itertools.count(start_update):
        # ...
        # ... (LR annealing logic) ...

        for step in range(0, num_steps):
            global_step += num_envs
            obs_buffer[step] = next_obs
            dones_buffer[step] = next_done
            
            # Alphabetical Slicing: [mask (192), obs (139)]
            current_mask = next_obs[:, :mask_size]
            current_obs = next_obs[:, mask_size : mask_size + obs_size]

            with torch.no_grad():
                action, logprob, _, value, _ = agent.get_action_and_value(current_obs, mask=current_mask)
                values_buffer[step] = value.flatten()
            
            actions_buffer[step] = action
            logprobs_buffer[step] = logprob

            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            
            # TRACK UNIVERSAL METRICS
            episode_rewards += reward
            episode_lengths += 1
            
            for i, d in enumerate(torch.logical_or(torch.Tensor(terminated), torch.Tensor(truncated))):
                if d:
                    # Log basics
                    writer.add_scalar("charts/episodic_return", episode_rewards[i], global_step)
                    writer.add_scalar("charts/episodic_length", episode_lengths[i], global_step)
                    
                    recent_returns.append(episode_rewards[i])
                    recent_lengths.append(episode_lengths[i])
                    
                    # Log lines cleared (Universal metric)
                    if "lines_cleared" in infos:
                        line_val = infos["lines_cleared"][i]
                        writer.add_scalar("charts/total_lines_cleared", line_val, global_step)
                        recent_lines.append(line_val)
                    elif "final_info" in infos:
                        final_info = infos["final_info"][i]
                        if final_info is not None and "lines_cleared" in final_info:
                            line_val = final_info["lines_cleared"]
                            writer.add_scalar("charts/total_lines_cleared", line_val, global_step)
                            recent_lines.append(line_val)

                    # Keep only last 100 episodes for console averaging
                    if len(recent_returns) > 100: recent_returns.pop(0)
                    if len(recent_lengths) > 100: recent_lengths.pop(0)
                    if len(recent_lines) > 100: recent_lines.pop(0)

                    episode_rewards[i] = 0
                    episode_lengths[i] = 0
                    total_episodes += 1

            next_done = torch.logical_or(torch.Tensor(terminated), torch.Tensor(truncated)).to(device)
            next_obs = torch.Tensor(next_obs).to(device)
            rewards_buffer[step] = torch.tensor(reward).to(device)

        # bootstrap value if not done
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

        # flatten the batch
        b_obs = obs_buffer.reshape((-1,) + obs_shape)
        b_logprobs = logprobs_buffer.reshape(-1)
        b_actions = actions_buffer.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buffer.reshape(-1)
        
        b_current_obs = b_obs[:, mask_size : mask_size + obs_size]
        b_current_mask = b_obs[:, :mask_size]

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
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

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Logging
        current_time = time.time() - start_time
        sps = int(global_step / current_time)
        eps = int(total_episodes / current_time)
        
        avg_ret = np.mean(recent_returns) if recent_returns else 0
        avg_len = np.mean(recent_lengths) if recent_lengths else 0
        avg_line = np.mean(recent_lines) if recent_lines else 0

        print(f"Update {update} | SPS: {sps} | EPS: {eps} | Ret: {avg_ret:.2f} | Len: {avg_len:.1f} | Lines: {avg_line:.1f}")
        
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("charts/EPS", eps, global_step)
        writer.add_scalar("charts/avg_reward", rewards_buffer.mean().item(), global_step)

        if update == 1 or update % 50 == 0:
            checkpoint_path = f"checkpoints/{run_name}/update_{update}.pt"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                "agent_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "update": update,
                "global_step": global_step,
            }, checkpoint_path)

    envs.close()
    writer.close()

if __name__ == "__main__":
    main()
