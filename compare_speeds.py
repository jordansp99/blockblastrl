import time
import torch
import env
import pufferlib
import pufferlib.vector
from torch_env import TorchBlockBlastEnv

def make_env(**kwargs):
    return pufferlib.emulation.GymnasiumPufferEnv(env_creator=lambda: env.BlockBlastEnv(), **kwargs)

if __name__ == "__main__":
    num_envs = 16384
    steps = 100
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Comparing environments with {num_envs} parallel instances for {steps} steps.")
    print(f"PyTorch Device: {device}\n")
    
    # 1. PUFFERLIB SERIAL (Current Setup)
    print("--- 1. PufferLib (C Library + Serial) ---")
    try:
        vec_envs = pufferlib.vector.make(
            make_env, 
            num_envs=num_envs, 
            backend=pufferlib.vector.Serial
        )
        obs, _ = vec_envs.reset()
        
        # Warmup
        mask = obs[:, :192]
        actions = torch.multinomial(torch.tensor(mask).float(), 1).squeeze(-1).numpy()
        vec_envs.step(actions)
        
        start_run = time.time()
        for _ in range(steps):
            mask = obs[:, :192]
            # Select random valid action
            actions = torch.multinomial(torch.tensor(mask).float(), 1).squeeze(-1).numpy()
            obs, reward, terminated, truncated, _ = vec_envs.step(actions)
            
        puffer_time = time.time() - start_run
        puffer_sps = int((num_envs * steps) / puffer_time)
        print(f"Time: {puffer_time:.2f} seconds")
        print(f"Speed: {puffer_sps:,} SPS")
        vec_envs.close()
    except Exception as e:
        print(f"PufferLib Multiprocessing failed: {e}")

    # 2. PURE PYTORCH ENVIRONMENT
    print("\n--- 2. Pure PyTorch Tensorized Environment ---")
    torch_env = TorchBlockBlastEnv(num_envs=num_envs, device=device)
    obs = torch_env.reset()
    
    # Warmup
    mask = obs[:, :192]
    actions = torch.multinomial(mask, 1).squeeze(-1)
    torch_env.step(actions)
    
    start_run = time.time()
    for _ in range(steps):
        mask = obs[:, :192]
        actions = torch.multinomial(mask, 1).squeeze(-1)
        obs, reward, terminated, truncated, _ = torch_env.step(actions)
        
    torch_time = time.time() - start_run
    torch_sps = int((num_envs * steps) / torch_time)
    print(f"Time: {torch_time:.2f} seconds")
    print(f"Speed: {torch_sps:,} SPS")
    
    if 'puffer_sps' in locals() and puffer_sps > 0:
        if torch_sps > puffer_sps:
            print(f"\nResult: PyTorch is {torch_sps/puffer_sps:.1f}x FASTER!")
        else:
            print(f"\nResult: PufferLib is {puffer_sps/torch_sps:.1f}x FASTER!")