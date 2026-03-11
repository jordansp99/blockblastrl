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

class ChampionAgent(nn.Module):
    def __init__(self, obs_size=139):
        super().__init__()
        # Local view (Local details)
        self.local_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Global view (Rows, Columns, and Whole Board)
        self.global_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8), # Full board view
            nn.ReLU(),
            nn.Flatten()
        )
        self.row_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 8)), # Horizontal line detector
            nn.ReLU(),
            nn.Flatten()
        )
        self.col_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(8, 1)), # Vertical line detector
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Combined features (4096 + 32 + 128 + 128 + 75 = 4459)
        self.fc = nn.Sequential(
            nn.Linear(4459, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.actor = nn.Linear(256, 192)
        self.critic = nn.Linear(256, 1)

    def _get_hidden(self, x):
        board = x[:, :64].view(-1, 1, 8, 8).float()
        shapes = x[:, 64:139].float()
        l_feat = self.local_conv(board)
        g_feat = self.global_conv(board)
        r_feat = self.row_conv(board)
        c_feat = self.col_conv(board)
        combined = torch.cat([l_feat, g_feat, r_feat, c_feat, shapes], dim=1)
        return self.fc(combined)

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a checkpoint to load")
    parser.add_argument("--run-name", type=str, default=None, help="The name of the run (for TensorBoard)")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000_000, help="Total timesteps (default: 1B for 'forever')")
    parser.add_argument("--no-tensorboard", action="store_true", help="Don't launch tensorboard")
    args = parser.parse_args()

    # PPO CONFIGURATION
    num_envs = 128 
    num_steps = 256 
    total_timesteps = args.total_timesteps
    learning_rate = 3e-4 # More stable for PPO
    anneal_lr = True
    gamma = 0.99
    gae_lambda = 0.95
    num_minibatches = 4
    update_epochs = 4
    norm_adv = True
    clip_coef = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5
    
    batch_size = int(num_envs * num_steps)
    minibatch_size = int(batch_size // num_minibatches)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting PPO training on {device}...")
    
    envs = pufferlib.vector.make(make_env, num_envs=num_envs, backend=pufferlib.vector.Serial)
    agent = ChampionAgent().to(device)
    
    start_update = 1
    global_step = 0
    
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            checkpoint = torch.load(args.checkpoint, map_location=device)
            # Handle both old (state_dict only) and new (dict with metadata) checkpoints
            if isinstance(checkpoint, dict) and "agent_state_dict" in checkpoint:
                agent.load_state_dict(checkpoint["agent_state_dict"])
                start_update = checkpoint.get("update", 0) + 1
                global_step = checkpoint.get("global_step", 0)
                print(f"Loaded checkpoint: {args.checkpoint} (Resuming from update {start_update})")
            else:
                agent.load_state_dict(checkpoint)
                print(f"Loaded checkpoint: {args.checkpoint} (Legacy format)")
        else:
            print(f"Error: Checkpoint {args.checkpoint} not found.")
            return

    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    
    # Use existing run_name if provided to continue TensorBoard logs
    run_name = args.run_name if args.run_name else f"PPO_MARATHON_{int(time.time())}"
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
    
    total_episodes = 0
    start_time = time.time()
    num_updates = total_timesteps // batch_size
    
    for update in itertools.count(start_update):
        # Annealing the rate if instructed to do so.
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = max(frac * learning_rate, 0.0)
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, num_steps):
            global_step += num_envs
            obs_buffer[step] = next_obs
            dones_buffer[step] = next_done
            
            # Alphabetical Slicing: [mask (192), obs (139)]
            current_mask = next_obs[:, :mask_size]
            current_obs = next_obs[:, mask_size : mask_size + obs_size]

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(current_obs, mask=current_mask)
                values_buffer[step] = value.flatten()
            
            actions_buffer[step] = action
            logprobs_buffer[step] = logprob
            
            next_obs, reward, terminated, truncated, _ = envs.step(action.cpu().numpy())
            next_done = torch.logical_or(torch.Tensor(terminated), torch.Tensor(truncated)).to(device)
            total_episodes += int(next_done.sum().item())
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

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
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
        avg_reward = float(rewards_buffer.mean().item())
        print(f"Update {update} | SPS: {sps} | EPS: {eps} | Reward: {avg_reward:.2f} | Var: {explained_var:.2f}")
        
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("charts/EPS", eps, global_step)
        writer.add_scalar("charts/avg_reward", avg_reward, global_step)

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
