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
import argparse

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

    def get_action(self, x, mask=None, lstm_state=None, deterministic=True):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(0)
            
        hidden, next_lstm_state = self._get_hidden(x, lstm_state)
        logits = self.actor(hidden)
        
        if mask is not None:
            logits = logits + (mask == 0) * -1e9
            
        if deterministic:
            action = torch.argmax(logits, dim=1)
        else:
            probs = Categorical(logits=logits)
            action = probs.sample()
            
        return action.item(), next_lstm_state

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("seed", type=int, nargs="?", default=None, help="Optional seed")
    parser.add_argument("--stochastic", action="store_true", help="Use sampling instead of argmax")
    parser.add_argument("--mcts", type=int, default=0, help="Number of MCTS simulations per move (0 to disable)")
    
    # Architecture arguments
    parser.add_argument("--arch", type=str, default="cnn", choices=["cnn", "mlp"], help="Architecture type")
    parser.add_argument("--fc-layers", type=int, nargs="+", default=[512, 512, 256], help="FC layer dimensions")
    parser.add_argument("--cnn-channels", type=int, nargs=2, default=[32, 64], help="CNN channels")
    parser.add_argument("--lstm", type=int, default=0, help="LSTM hidden size (0 to disable)")
    parser.add_argument("--transformer-layers", type=int, default=0, help="Transformer layers")
    parser.add_argument("--transformer-heads", type=int, default=4, help="Transformer heads")
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "gelu"], help="Activation function")

    args = parser.parse_args()
        
    checkpoint_path = args.checkpoint
    seed = args.seed
    num_sims = args.mcts
    
    # Auto-load config if it exists
    import json
    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(checkpoint_dir, "config.json")
    
    if os.path.exists(config_path):
        print(f"Loading auto-config from {config_path}")
        with open(config_path, "r") as f:
            config = json.load(f)
        if "--arch" not in sys.argv: args.arch = config.get("arch", args.arch)
        if "--fc-layers" not in sys.argv: args.fc_layers = config.get("fc_layers", args.fc_layers)
        if "--cnn-channels" not in sys.argv: args.cnn_channels = config.get("cnn_channels", args.cnn_channels)
        if "--lstm" not in sys.argv: args.lstm = config.get("lstm", args.lstm)
        if "--transformer-layers" not in sys.argv: args.transformer_layers = config.get("transformer_layers", args.transformer_layers)
        if "--transformer-heads" not in sys.argv: args.transformer_heads = config.get("transformer_heads", args.transformer_heads)
        if "--activation" not in sys.argv: args.activation = config.get("activation", args.activation)
    else:
        print("No config.json found, using defaults or CLI arguments.")

    # Initialize environment
    import pufferlib.emulation
    blockblast_env = env.BlockBlastEnv(render_mode="human", seed=seed)
    puffer_env = pufferlib.emulation.GymnasiumPufferEnv(
        env_creator=lambda: blockblast_env
    )
    
    obs_size = 139
    action_size = 192
    
    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device} for playback.")
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
    agent = FlexibleAgent(
        obs_size=obs_size, 
        action_size=action_size, 
        arch_type=args.arch,
        fc_layers=args.fc_layers,
        cnn_channels=args.cnn_channels,
        lstm_hidden=args.lstm,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        activation=args.activation
    ).to(device)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict):
            if "agent_state_dict" in checkpoint:
                agent.load_state_dict(checkpoint["agent_state_dict"])
                print(f"Loaded checkpoint: {checkpoint_path} (from update {checkpoint.get('update')})")
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
                agent.load_state_dict(new_state_dict)
                print(f"Loaded checkpoint: {checkpoint_path} (from remote format)")
            else:
                agent.load_state_dict(checkpoint)
                print(f"Loaded checkpoint: {checkpoint_path} (Unknown dict format)")
        else:
            agent.load_state_dict(checkpoint)
            print(f"Loaded checkpoint: {checkpoint_path} (Legacy format)")
    else:
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        sys.exit(1)
        
    agent.eval()
    
    # Setup MCTS if requested
    mcts = None
    if num_sims > 0:
        import mcts as mcts_lib
        # Based on blockblast_lib.c, the state size is 1024 to be safe.
        mcts = mcts_lib.MCTSEngine(agent, device, blockblast_env.lib, 1024)
        print(f"MCTS enabled with {num_sims} simulations per move.")

    obs, info = puffer_env.reset()
    done = False
    
    lstm_state = None
    mask_size, obs_size = 192, 139
    
    print("AI is now playing...")
    
    while not done:
        if mcts:
            # Use MCTS to find the best action
            action = mcts.search(blockblast_env.state_ptr, num_simulations=num_sims)
        else:
            # Standard Reactive Policy
            obs_flat = obs.flatten()
            tensor_mask = torch.Tensor(obs_flat[:mask_size]).unsqueeze(0).to(device)
            tensor_obs = torch.Tensor(obs_flat[mask_size : mask_size + obs_size]).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action, lstm_state = agent.get_action(
                    tensor_obs, 
                    mask=tensor_mask, 
                    lstm_state=lstm_state,
                    deterministic=not args.stochastic
                )
            
        obs, reward, terminated, truncated, info = puffer_env.step(action)
        done = terminated or truncated
        time.sleep(0.5) 
        
    print("Game Over!")
    time.sleep(5.0)
    puffer_env.close()

if __name__ == "__main__":
    main()