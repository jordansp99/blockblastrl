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
        
        # Combined features
        self.fc = nn.Sequential(
            nn.Linear(4096 + 32 + 128 + 128 + 75, 512),
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
        shapes = x[:, 64:].float()
        l_feat = self.local_conv(board)
        g_feat = self.global_conv(board)
        r_feat = self.row_conv(board)
        c_feat = self.col_conv(board)
        combined = torch.cat([l_feat, g_feat, r_feat, c_feat, shapes], dim=1)
        return self.fc(combined)

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
    # AGGRESSIVE MARATHON CONFIGURATION
    num_envs = 128 
    num_steps = 512 # Longer memory for line completion awareness
    total_timesteps = 50000000 
    learning_rate = 1e-3 # Fast learning
    ent_coef = 0.01 # Focus on exploitation
    gamma = 0.999
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Starting Strategic Marathon on {device}...")
    
    envs = pufferlib.vector.make(make_env, num_envs=num_envs, backend=pufferlib.vector.Serial)
    agent = ChampionAgent().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    
    run_name = f"STRATEGIC_MARATHON_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    
    obs_size, mask_size = 139, 192
    obs_shape = envs.single_observation_space.shape
    
    obs_buffer = torch.zeros((num_steps, num_envs) + obs_shape).to(device)
    actions_buffer = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
    logprobs_buffer = torch.zeros((num_steps, num_envs)).to(device)
    rewards_buffer = torch.zeros((num_steps, num_envs)).to(device)
    values_buffer = torch.zeros((num_steps, num_envs)).to(device)
    masks_buffer = torch.zeros((num_steps, num_envs, mask_size)).to(device)
    
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    
    global_step = 0
    start_time = time.time()
    num_iterations = total_timesteps // (num_envs * num_steps)
    
    for iteration in range(1, num_iterations + 1):
        for step in range(0, num_steps):
            global_step += num_envs
            obs_buffer[step] = next_obs
            
            # Alphabetical Slicing
            current_mask = next_obs[:, :mask_size]
            current_obs = next_obs[:, mask_size : mask_size + obs_size]
            masks_buffer[step] = current_mask

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(current_obs, mask=current_mask)
                values_buffer[step] = value.flatten()
            
            actions_buffer[step] = action
            logprobs_buffer[step] = logprob
            
            next_obs, reward, _, _, _ = envs.step(action.cpu().numpy())
            next_obs = torch.Tensor(next_obs).to(device)
            rewards_buffer[step] = torch.tensor(reward).to(device)

        # Optimization Step
        b_obs = obs_buffer.reshape((-1,) + obs_shape)
        b_logprobs = logprobs_buffer.reshape(-1)
        b_actions = actions_buffer.reshape((-1,) + envs.single_action_space.shape)
        b_returns = rewards_buffer.reshape(-1)
        b_values = values_buffer.reshape(-1)
        b_masks = masks_buffer.reshape((-1, mask_size))
        
        b_current_obs = b_obs[:, mask_size : mask_size + obs_size]
        b_current_mask = b_obs[:, :mask_size]
        
        _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_current_obs, b_actions.long(), mask=b_current_mask)
        
        pg_loss = -newlogprob.mean()
        v_loss = ((newvalue.view(-1) - b_returns) ** 2).mean() * 0.5
        ent_loss = entropy.mean()
        loss = pg_loss + v_loss - ent_coef * ent_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Logging
        avg_reward = rewards_buffer.mean().item()
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/avg_reward", avg_reward, iteration)
        writer.add_scalar("charts/SPS", sps, iteration)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), iteration)
        writer.add_scalar("losses/value_loss", v_loss.item(), iteration)
        writer.add_scalar("losses/entropy", ent_loss.item(), iteration)
        
        if iteration % 100 == 0 or iteration == num_iterations:
            checkpoint_path = f"checkpoints/{run_name}/iter_{iteration}.pt"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(agent.state_dict(), checkpoint_path)
            print(f"Iter {iteration}/{num_iterations} | Reward: {avg_reward:.2f} | SPS: {sps}")

    envs.close()
    writer.close()

if __name__ == "__main__":
    main()
