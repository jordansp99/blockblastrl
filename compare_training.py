import time
import torch
import torch.nn as nn
import torch.optim as optim
import env
import pufferlib
import pufferlib.vector
from torch_env import TorchBlockBlastEnv

class SimpleAgent(nn.Module):
    def __init__(self, action_size=192):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(139, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.actor = nn.Linear(256, action_size)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        h = self.fc(x)
        return self.actor(h), self.critic(h)

def make_env(**kwargs):
    return pufferlib.emulation.GymnasiumPufferEnv(env_creator=lambda: env.BlockBlastEnv(), **kwargs)

if __name__ == "__main__":
    num_envs = 4096
    steps_per_update = 128
    num_updates = 5
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Optimization for NVIDIA: Allow CPU and GPU to work in parallel
        pin_memory = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        pin_memory = False
    else:
        device = torch.device("cpu")
        pin_memory = False
        
    print(f"REAL-WORLD TRAINING BENCHMARK")
    print(f"Envs: {num_envs} | Steps/Update: {steps_per_update}")
    print(f"Device: {device}\n")

    # 1. PUFFERLIB
    print("--- 1. PufferLib (CPU Env -> GPU Network) ---")
    try:
        vec_envs = pufferlib.vector.make(make_env, num_envs=num_envs, backend=pufferlib.vector.Serial)
        agent = SimpleAgent().to(device)
        optimizer = optim.Adam(agent.parameters(), lr=1e-3)
        obs, _ = vec_envs.reset()
        
        start_run = time.time()
        for u in range(num_updates):
            for _ in range(steps_per_update):
                # Data move with non_blocking if on CUDA
                obs_tensor = torch.tensor(obs[:, 192 : 192 + 139], dtype=torch.float32, device=device, non_blocking=pin_memory)
                mask = torch.tensor(obs[:, :192], dtype=torch.float32, device=device, non_blocking=pin_memory)
                with torch.no_grad():
                    logits, _ = agent(obs_tensor)
                    logits = logits + (mask == 0) * -1e9
                    probs = torch.softmax(logits, dim=1)
                    actions = torch.multinomial(probs, 1).squeeze(-1)
                obs, reward, terminated, truncated, _ = vec_envs.step(actions.cpu().numpy())
            
            optimizer.zero_grad()
            logits, values = agent(obs_tensor)
            loss = logits.mean() + values.mean()
            loss.backward()
            optimizer.step()
            print(f"  Update {u+1}/{num_updates} complete...")
            
        puffer_time = time.time() - start_run
        puffer_sps = int((num_envs * steps_per_update * num_updates) / puffer_time)
        print(f"Total Time: {puffer_time:.2f}s | Speed: {puffer_sps:,} SPS")
        vec_envs.close()
    except Exception as e:
        print(f"PufferLib Benchmark Failed: {e}")

    # 2. PURE PYTORCH
    print("\n--- 2. Pure PyTorch (Everything on GPU) ---")
    try:
        torch_env = TorchBlockBlastEnv(num_envs=num_envs, device=device)
        agent = SimpleAgent().to(device)
        optimizer = optim.Adam(agent.parameters(), lr=1e-3)
        obs = torch_env.reset()
        
        start_run = time.time()
        for u in range(num_updates):
            for _ in range(steps_per_update):
                # Correct Slicing: [mask (192), obs (139)]
                mask = obs[:, :192]
                obs_data = obs[:, 192 : 192 + 139]
                with torch.no_grad():
                    logits, _ = agent(obs_data)
                    logits = logits + (mask == 0) * -1e9
                    probs = torch.softmax(logits, dim=1)
                    actions = torch.multinomial(probs, 1).squeeze(-1)
                obs, rewards, dones, _, _ = torch_env.step(actions)
            
            optimizer.zero_grad()
            logits, values = agent(obs_data)
            loss = logits.mean() + values.mean()
            loss.backward()
            optimizer.step()
            print(f"  Update {u+1}/{num_updates} complete...")
            
        torch_time = time.time() - start_run
        torch_sps = int((num_envs * steps_per_update * num_updates) / torch_time)
        print(f"Total Time: {torch_time:.2f}s | Speed: {torch_sps:,} SPS")
    except Exception as e:
        print(f"PyTorch Benchmark Failed: {e}")

    if 'puffer_sps' in locals() and 'torch_sps' in locals():
        if torch_sps > puffer_sps:
            print(f"\nWINNER: Pure PyTorch is {torch_sps/puffer_sps:.1f}x FASTER!")
        else:
            print(f"\nWINNER: PufferLib is {puffer_sps/torch_sps:.1f}x FASTER!")