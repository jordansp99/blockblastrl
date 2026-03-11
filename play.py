import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym")
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np
import env
import time
import os
import sys

# The Agent model needs to match the structure in train.py
class Agent(nn.Module):
    def __init__(self, obs_size, action_size, arch="cnn"):
        super().__init__()
        # Local view
        self.local_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Global view
        self.global_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8),
            nn.ReLU(),
            nn.Flatten()
        )
        self.row_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 8)),
            nn.ReLU(),
            nn.Flatten()
        )
        self.col_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(8, 1)),
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
        self.actor = nn.Linear(256, action_size)
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

    def get_action(self, x, mask=None):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(0)
            
        hidden = self._get_hidden(x)
        logits = self.actor(hidden)
        
        if mask is not None:
            logits = logits + (mask == 0) * -1e9
            
        probs = Categorical(logits=logits)
        return probs.sample().item()

def main():
    if len(sys.argv) < 2:
        print("Usage: python play.py <checkpoint_path> [seed]")
        sys.exit(1)
        
    checkpoint_path = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else None
    arch = "cnn" # Default to the Champion CNN architecture
    
    # Initialize environment via PufferLib wrapper
    import pufferlib.emulation
    puffer_env = pufferlib.emulation.GymnasiumPufferEnv(
        env_creator=lambda: env.BlockBlastEnv(render_mode="human", seed=seed)
    )
    
    obs_size = 139
    action_size = 192
    
    # Detect device (CUDA, MPS, or CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device} for playback.")
    agent = Agent(obs_size, action_size, arch=arch).to(device)
    
    if os.path.exists(checkpoint_path):
        agent.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        sys.exit(1)
        
    agent.eval()
    obs, info = puffer_env.reset()
    done = False
    
    print("AI is now playing (Slow Speed)...")
    
    while not done:
        # Correct Alphabetical Slicing: [mask (192), obs (139)]
        obs_flat = obs.flatten()
        tensor_mask = torch.Tensor(obs_flat[:192]).unsqueeze(0).to(device)
        tensor_obs = torch.Tensor(obs_flat[192 : 192 + 139]).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action = agent.get_action(tensor_obs, mask=tensor_mask)
            
        obs, reward, terminated, truncated, info = puffer_env.step(action)
        done = terminated or truncated
        
        # SLOW SPEED: 1 second per move
        time.sleep(1.0)
        
    print("Game Over! Board is full or no valid moves left.")
    # HANG ON GAMEOVER: Wait 5 seconds before closing
    time.sleep(5.0)
    puffer_env.close()

if __name__ == "__main__":
    main()
