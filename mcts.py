import torch
import numpy as np
import math
from ctypes import *

class MCTSNode:
    def __init__(self, action_id=None, parent=None, prior=0):
        self.action_id = action_id
        self.parent = parent
        self.prior = prior
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0

    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0

class MCTSEngine:
    """Standard Single-state MCTS for play.py"""
    def __init__(self, agent, device, lib, c_struct_size):
        self.agent = agent
        self.device = device
        self.lib = lib
        self.c_struct_size = c_struct_size
        self.c_puct = 1.4

    def search(self, root_state_ptr, num_simulations=100):
        root = MCTSNode()
        mask_buf = (c_int * 192)()
        obs_buf = (c_int * 139)()
        
        for _ in range(num_simulations):
            node = root
            temp_state = create_string_buffer(self.c_struct_size)
            self.lib.get_game_state(root_state_ptr, temp_state)
            
            accum_reward = 0
            depth = 0
            done = c_bool(False)
            
            while node.children and depth < 3:
                node = self._select_child(node)
                reward = c_float(0.0)
                self.lib.step_game(temp_state, int(node.action_id), byref(reward), byref(done))
                accum_reward += reward.value
                depth += 1
                if done.value: break
            
            self.lib.get_observation(temp_state, obs_buf)
            obs_tensor = torch.from_numpy(np.frombuffer(obs_buf, dtype=np.int32).copy()).float().to(self.device).unsqueeze(0)
            
            with torch.no_grad():
                model = self.agent._orig_mod if hasattr(self.agent, "_orig_mod") else self.agent
                hidden = model._get_hidden(obs_tensor)
                value = model.critic(hidden).item() + (accum_reward / 100.0)
                
                if not done.value and depth < 3:
                    self.lib.get_action_mask(temp_state, mask_buf)
                    mask = torch.from_numpy(np.frombuffer(mask_buf, dtype=np.int32).copy()).to(self.device)
                    logits = model.actor(hidden) + (mask == 0) * -1e9
                    priors = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                    for idx in np.where(mask.cpu().numpy() > 0)[0]:
                        if idx not in node.children:
                            node.children[idx] = MCTSNode(action_id=idx, parent=node, prior=priors[idx])
            
            while node:
                node.visit_count += 1
                node.value_sum += value
                node = node.parent
                
        return max(root.children.items(), key=lambda x: x[1].visit_count)[0]

    def _select_child(self, node):
        total_v = sum(c.visit_count for c in node.children.values())
        sqrt_v = math.sqrt(total_v) if total_v > 0 else 0
        return max(node.children.values(), key=lambda c: c.value() + self.c_puct * c.prior * (sqrt_v / (1 + c.visit_count)))

class BatchedMCTSEngine:
    """High-performance Vectorized MCTS for train.py"""
    def __init__(self, agent, device, lib, c_struct_size, num_envs):
        self.agent = agent
        self.device = device
        self.lib = lib
        self.c_struct_size = c_struct_size
        self.num_envs = num_envs
        self.c_puct = 1.4
        
        # CONTIGUOUS BUFFERS FOR FAST C-HANDLING
        self.obs_buffer = np.zeros(num_envs * 139, dtype=np.int32)
        self.mask_buffer = np.zeros(num_envs * 192, dtype=np.int32)
        self.reward_buffer = np.zeros(num_envs, dtype=np.float32)
        self.done_buffer = np.zeros(num_envs, dtype=np.bool_)
        self.action_buffer = np.zeros(num_envs, dtype=np.int32)
        
        # Temp GameState pointers
        self.temp_states = [create_string_buffer(c_struct_size) for _ in range(num_envs)]
        self.temp_state_ptrs = (c_void_p * num_envs)(*[cast(s, c_void_p) for s in self.temp_states])

    def search(self, root_ptrs, num_simulations=50):
        roots = [MCTSNode() for _ in range(self.num_envs)]
        model = self.agent._orig_mod if hasattr(self.agent, "_orig_mod") else self.agent
        
        c_root_ptrs = (c_void_p * self.num_envs)(*root_ptrs)

        for _ in range(num_simulations):
            nodes = []
            accum_rewards = np.zeros(self.num_envs, dtype=np.float32)
            active_mask = np.ones(self.num_envs, dtype=bool)
            depths = np.zeros(self.num_envs, dtype=int)
            
            # 1. BATCH COPY STATE
            self.lib.copy_game_state_batch(c_root_ptrs, self.temp_state_ptrs, self.num_envs)
            
            # 2. SELECTION (Still sequential but minimal work)
            for i in range(self.num_envs):
                node = roots[i]
                while node.children and depths[i] < 3:
                    total_v = sum(c.visit_count for c in node.children.values())
                    sqrt_v = math.sqrt(total_v) if total_v > 0 else 0
                    node = max(node.children.values(), key=lambda c: c.value() + self.c_puct * c.prior * (sqrt_v / (1 + c.visit_count)))
                    
                    # Store action for batch stepping
                    self.action_buffer[i] = int(node.action_id)
                    
                    # Single step in C (we could batch this too if we track paths)
                    rew = c_float(0)
                    d = c_bool(False)
                    self.lib.step_game(self.temp_state_ptrs[i], self.action_buffer[i], byref(rew), byref(d))
                    accum_rewards[i] += rew.value
                    depths[i] += 1
                    if d.value: break
                nodes.append(node)

            # 3. BATCH OBSERVATION (Single C Call)
            self.lib.get_observation_batch(self.temp_state_ptrs, self.obs_buffer.ctypes.data_as(POINTER(c_int)), self.num_envs)
            
            # 4. BATCH EVALUATION (GPU)
            obs_tensor = torch.from_numpy(self.obs_buffer).view(self.num_envs, 139).float().to(self.device)
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    hidden = model._get_hidden(obs_tensor)
                    values = model.critic(hidden).flatten().cpu().numpy()
                    all_logits = model.actor(hidden)
            
            final_vals = values + (accum_rewards / 100.0)

            # 5. BATCH MASKING (Single C Call)
            self.lib.get_action_mask_batch(self.temp_state_ptrs, self.mask_buffer.ctypes.data_as(POINTER(c_int)), self.num_envs)
            masks_tensor = torch.from_numpy(self.mask_buffer).view(self.num_envs, 192).to(self.device)
            
            # 6. BATCH PRIORS
            priors_batch = torch.softmax(all_logits + (masks_tensor == 0) * -1e9, dim=1).cpu().numpy()

            # 7. EXPANSION & BACKPROP
            for i in range(self.num_envs):
                node = nodes[i]
                if depths[i] < 3:
                    legal = np.where(self.mask_buffer.reshape(-1, 192)[i] > 0)[0]
                    for idx in legal:
                        if idx not in node.children:
                            node.children[idx] = MCTSNode(action_id=idx, parent=node, prior=priors_batch[i][idx])
                
                val = final_vals[i]
                while node:
                    node.visit_count += 1
                    node.value_sum += val
                    node = node.parent

        # EXTRACT RESULTS
        best_actions = np.zeros(self.num_envs, dtype=np.int32)
        target_dists = np.zeros((self.num_envs, 192), dtype=np.float32)
        for i in range(self.num_envs):
            dist = np.zeros(192)
            for act, child in roots[i].children.items():
                dist[act] = child.visit_count
            if dist.sum() > 0:
                dist /= dist.sum()
                best_actions[i] = np.argmax(dist)
            target_dists[i] = dist

        return best_actions, target_dists
