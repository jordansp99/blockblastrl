import torch
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
        self.actor = nn.Linear(256, action_size)
        self.critic = nn.Linear(256, 1)

    def _get_hidden(self, x):
        if self.arch == "cnn":
            board = x[:, :64].view(-1, 1, 8, 8).float()
            shapes = x[:, 64:139].float()
            board_features = self.conv_net(board)
            return self.fc(torch.cat([board_features, shapes], dim=1))
        else:
            return self.fc(x[:, :139].float())

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
    if len(sys.argv) < 3:
        print("Usage: python play.py <checkpoint_path> <arch: cnn/mlp>")
        sys.exit(1)
        
    checkpoint_path = sys.argv[1]
    arch = sys.argv[2]
    
    # Initialize environment via PufferLib wrapper
    import pufferlib.emulation
    puffer_env = pufferlib.emulation.GymnasiumPufferEnv(
        env_creator=lambda: env.BlockBlastEnv(render_mode="human")
    )
    
    obs_size = 139
    action_size = 192
    
    device = torch.device("cpu")
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
        # PufferLib flattens: [obs (139), mask (192)]
        tensor_obs = torch.Tensor(obs[:139])
        tensor_mask = torch.Tensor(obs[-192:])
        
        with torch.no_grad():
            action = agent.get_action(tensor_obs, mask=tensor_mask)
            
        obs, reward, terminated, truncated, info = puffer_env.step(action)
        done = terminated or truncated
        
        # SLOW SPEED: 1 second per move
        time.sleep(1.0)
        
    print("Game Over!")
    puffer_env.close()

if __name__ == "__main__":
    main()
