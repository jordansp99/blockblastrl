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
        print("Usage: python play.py <checkpoint_path> [seed] [--mcts <sims>]")
        sys.exit(1)
        
    checkpoint_path = sys.argv[1]
    seed = None
    num_sims = 0
    
    # Parse simple args
    for i in range(2, len(sys.argv)):
        if sys.argv[i] == "--mcts":
            num_sims = int(sys.argv[i+1])
        elif sys.argv[i].isdigit():
            seed = int(sys.argv[i])
            
    arch = "cnn"
    
    # Initialize environment
    import pufferlib.emulation
    blockblast_env = env.BlockBlastEnv(render_mode="human", seed=seed)
    puffer_env = pufferlib.emulation.GymnasiumPufferEnv(
        env_creator=lambda: blockblast_env
    )
    
    obs_size = 139
    action_size = 192
    
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for playback.")
    agent = Agent(obs_size, action_size, arch=arch).to(device)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            agent.load_state_dict(new_state_dict)
        else:
            agent.load_state_dict(checkpoint)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        sys.exit(1)
        
    agent.eval()
    
    # Setup MCTS if requested
    mcts = None
    if num_sims > 0:
        import mcts as mcts_lib
        from ctypes import sizeof
        # We need to know the size of the GameState struct
        # Based on blockblast_lib.c:
        # board[8][8] (64 ints), board_colors[8][8] (64 ints), 
        # current_shapes[3] (3 ints), shape_active[3] (3 bools), 
        # score (1 int), game_over (1 bool)
        # Note: Bools in C are often 1 byte, but ints are 4. 
        # A safer way is to just use a large enough buffer or define the struct in ctypes.
        # Looking at the struct, it's roughly (64+64+3+1)*4 + 4 = 532 bytes.
        # Let's use 1024 to be safe.
        mcts = mcts_lib.MCTSEngine(agent, device, blockblast_env.lib, 1024)
        print(f"MCTS enabled with {num_sims} simulations per move.")

    obs, info = puffer_env.reset()
    done = False
    
    print("AI is now playing...")
    
    mask_size, obs_size = 192, 139
    
    while not done:
        if mcts:
            # Use MCTS to find the best action
            action = mcts.search(blockblast_env.state_ptr, num_simulations=num_sims)
        else:
            # Standard Reactive Policy
            obs_flat = torch.Tensor(obs.flatten()).to(device)
            current_mask = obs_flat[:mask_size].unsqueeze(0)
            current_obs = obs_flat[mask_size : mask_size + obs_size].unsqueeze(0)
            with torch.no_grad():
                action = agent.get_action(current_obs, mask=current_mask)
            
        obs, reward, terminated, truncated, info = puffer_env.step(action)
        done = terminated or truncated
        time.sleep(0.5) # Slightly faster than before
        
    print("Game Over!")
    time.sleep(5.0)
    puffer_env.close()

if __name__ == "__main__":
    main()
