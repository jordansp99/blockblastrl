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
    parser.add_argument("--num-envs", type=int, default=1024, help="Number of parallel environments")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a checkpoint to load")
    parser.add_argument("--run-name", type=str, default=None, help="The name of the run (for TensorBoard)")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000_000, help="Total timesteps (default: 1B for 'forever')")
    parser.add_argument("--no-tensorboard", action="store_true", help="Don't launch tensorboard")
    parser.add_argument("--mcts-sims", type=int, default=0, help="Number of MCTS simulations per step (0 to disable)")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    args = parser.parse_args()

    # PPO CONFIGURATION
    num_envs = args.num_envs
    num_steps = 128
    total_timesteps = args.total_timesteps
    learning_rate = 3e-4 
    anneal_lr = True
    gamma = 0.99
    gae_lambda = 0.95
    num_minibatches = 8
    update_epochs = 4
    norm_adv = True
    clip_coef = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5
    mcts_sims = args.mcts_sims
    
    batch_size = int(num_envs * num_steps)
    minibatch_size = int(batch_size // num_minibatches)
    
    import psutil
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting PPO training on {device} with {num_envs} envs...")
    
    # MCTS training requires Serial backend to access state pointers
    backend = pufferlib.vector.Multiprocessing if mcts_sims == 0 else pufferlib.vector.Serial
    
    if mcts_sims == 0:
        num_workers = psutil.cpu_count(logical=False)
        while num_envs % num_workers != 0:
            num_workers -= 1
    else:
        num_workers = 1 # Serial
        print("MCTS-Guided Training Enabled. Using Serial backend for state access.")

    envs = pufferlib.vector.make(
        make_env, 
        num_envs=num_envs, 
        num_workers=num_workers,
        backend=backend
    )
    
    agent = ChampionAgent().to(device)
    
    global_step = 0
    total_episodes = 0
    start_update = 1
    scaler = torch.amp.GradScaler('cuda')
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            checkpoint = torch.load(args.checkpoint, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                # Strip _orig_mod. prefix if it exists
                state_dict = checkpoint["model_state_dict"]
                new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
                agent.load_state_dict(new_state_dict)
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
                global_step = checkpoint["global_step"]
                start_update = checkpoint["update"] + 1
                print(f"Resumed from checkpoint: {args.checkpoint} at update {checkpoint['update']}")
            else:
                agent.load_state_dict(checkpoint)
                print(f"Loaded weights from checkpoint: {args.checkpoint}")
        else:
            print(f"Error: Checkpoint {args.checkpoint} not found.")
            return

    try:
        if not args.no_compile:
            print("Compiling model for speed (this can take 2-5 minutes on the first run)...")
            agent = torch.compile(agent)
        else:
            print("Torch compile disabled.")
    except Exception as e:
        print(f"Torch compile failed: {e}")

    # Use existing run_name if provided to continue TensorBoard logs
    run_name = args.run_name if args.run_name else f"PPO_MARATHON_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    
    if not args.no_tensorboard:
        print(f"Launching TensorBoard for {run_name}...")
        subprocess.Popen(["tensorboard", "--logdir=runs", "--port=6006"], 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
    
    obs_size, mask_size = 139, 192
    obs_shape = envs.single_observation_space.shape
    
    # Buffers
    obs_buffer = torch.zeros((num_steps, num_envs) + obs_shape).to(device)
    actions_buffer = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
    logprobs_buffer = torch.zeros((num_steps, num_envs)).to(device)
    rewards_buffer = torch.zeros((num_steps, num_envs)).to(device)
    dones_buffer = torch.zeros((num_steps, num_envs)).to(device)
    values_buffer = torch.zeros((num_steps, num_envs)).to(device)
    # Target distribution for MCTS distillation
    mcts_targets_buffer = torch.zeros((num_steps, num_envs, 192)).to(device)
    
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(num_envs).to(device)
    
    # Initialize MCTS if enabled
    mcts_engine = None
    if mcts_sims > 0:
        import mcts
        # Access the underlying env's lib through the PufferLib wrapper
        c_lib = envs.envs[0].env.lib
        mcts_engine = mcts.BatchedMCTSEngine(agent, device, c_lib, 1024, num_envs)

    start_time = time.time()
    num_updates = total_timesteps // batch_size
    
    for update in itertools.count(start_update):
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = max(frac * learning_rate, 0.0)
            optimizer.param_groups[0]["lr"] = lrnow

        print(f"Starting rollout for Update {update}...")
        for step in range(0, num_steps):
            if step % 10 == 0:
                print(f"  Step {step}/{num_steps}...")
            global_step += num_envs
            obs_buffer[step] = next_obs
            dones_buffer[step] = next_done
            
            current_mask = next_obs[:, :mask_size]
            current_obs = next_obs[:, mask_size : mask_size + obs_size]

            if mcts_engine:
                # Use MCTS to find best action and target distribution
                # Access underlying env state_ptr
                state_ptrs = [e.env.state_ptr for e in envs.envs]
                action, target_dist = mcts_engine.search(state_ptrs, num_simulations=mcts_sims)
                action = torch.tensor(action).to(device)
                mcts_targets_buffer[step] = torch.tensor(target_dist).to(device)
                
                # Still need logprobs and values for PPO part
                with torch.no_grad():
                    _, logprob, _, value = agent.get_action_and_value(current_obs, action=action, mask=current_mask)
                    values_buffer[step] = value.flatten()
                    logprobs_buffer[step] = logprob
            else:
                # Standard PPO Rollout
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(current_obs, mask=current_mask)
                    values_buffer[step] = value.flatten()
                logprobs_buffer[step] = logprob
            
            actions_buffer[step] = action
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
        b_mcts_targets = mcts_targets_buffer.reshape(-1, 192)
        
        b_current_obs = b_obs[:, mask_size : mask_size + obs_size]
        b_current_mask = b_obs[:, :mask_size]

        b_inds = np.arange(batch_size)
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                with torch.amp.autocast('cuda'):
                    # PPO forward pass
                    new_action_id, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_current_obs[mb_inds], 
                        b_actions.long()[mb_inds], 
                        mask=b_current_mask[mb_inds]
                    )
                    
                    # Policy Loss (PPO)
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    mb_advantages = b_advantages[mb_inds]
                    if norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value Loss
                    v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()

                    # Distillation Loss (AlphaZero style)
                    # Train Actor to match MCTS visit distribution
                    distill_loss = 0
                    if mcts_sims > 0:
                        hidden = (agent._orig_mod if hasattr(agent, "_orig_mod") else agent)._get_hidden(b_current_obs[mb_inds])
                        logits = (agent._orig_mod if hasattr(agent, "_orig_mod") else agent).actor(hidden)
                        logits = logits + (b_current_mask[mb_inds] == 0) * -1e9
                        log_probs = torch.log_softmax(logits, dim=1)
                        distill_loss = -(b_mcts_targets[mb_inds] * log_probs).sum(dim=1).mean()

                    loss = pg_loss - ent_coef * entropy.mean() + v_loss * vf_coef + distill_loss

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        current_time = time.time() - start_time
        sps = int(global_step / current_time)
        avg_reward = float(rewards_buffer.mean().item())
        print(f"Update {update} | SPS: {sps} | Reward: {avg_reward:.2f} | Var: {explained_var:.2f}")
        
        writer.add_scalar("charts/avg_reward", avg_reward, global_step)
        writer.add_scalar("charts/SPS", sps, global_step)

        if update == 1 or update % 50 == 0:
            checkpoint_path = f"checkpoints/{run_name}/update_{update}.pt"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                "update": update,
                "global_step": global_step,
                "model_state_dict": (agent._orig_mod if hasattr(agent, "_orig_mod") else agent).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
            }, checkpoint_path)

    envs.close()
    writer.close()

if __name__ == "__main__":
    main()
