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
    def __init__(self, agent, device, lib, c_struct_size):
        self.agent = agent
        self.device = device
        self.lib = lib
        self.c_struct_size = c_struct_size
        self.c_puct = 1.4
        self.mask_size = 192
        self.obs_size = 139

    def search(self, root_state_ptr, num_simulations=100):
        # Single-state search (existing logic, updated for depth)
        root = MCTSNode()
        mask_buffer = (c_int * self.mask_size)()
        obs_buffer = (c_int * self.obs_size)()
        
        for _ in range(num_simulations):
            node = root
            temp_state_ptr = create_string_buffer(self.c_struct_size)
            self.lib.get_game_state(root_state_ptr, temp_state_ptr)
            
            accumulated_reward = 0
            search_depth = 0
            done = c_bool(False)
            
            while node.children and search_depth < 3:
                node = self._select_child(node)
                reward = c_float(0.0)
                self.lib.step_game(temp_state_ptr, int(node.action_id), byref(reward), byref(done))
                accumulated_reward += reward.value
                search_depth += 1
                if done.value: break
            
            self.lib.get_observation(temp_state_ptr, obs_buffer)
            obs = torch.Tensor(list(obs_buffer)).to(self.device)
            
            with torch.no_grad():
                model = self.agent._orig_mod if hasattr(self.agent, "_orig_mod") else self.agent
                hidden = model._get_hidden(obs.unsqueeze(0))
                value = model.critic(hidden).item() + (accumulated_reward / 100.0)
                
                if not done.value and search_depth < 3:
                    self.lib.get_action_mask(temp_state_ptr, mask_buffer)
                    mask = torch.Tensor(list(mask_buffer)).to(self.device)
                    logits = model.actor(hidden)
                    logits = logits + (mask == 0) * -1e9
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
        total_visits = sum(child.visit_count for child in node.children.values())
        sqrt_total = math.sqrt(total_visits) if total_visits > 0 else 0
        return max(node.children.values(), key=lambda c: c.value() + self.c_puct * c.prior * (sqrt_total / (1 + c.visit_count)))

class BatchedMCTSEngine:
    """High-performance MCTS that searches for many environments in parallel"""
    def __init__(self, agent, device, lib, c_struct_size, num_envs):
        self.agent = agent
        self.device = device
        self.lib = lib
        self.c_struct_size = c_struct_size
        self.num_envs = num_envs
        self.c_puct = 1.4
        
        # Pre-allocate buffers for speed
        self.mask_buffer = (c_int * 192)()
        self.obs_buffer = (c_int * 139)()

    def search(self, state_ptrs, num_simulations=50):
        roots = [MCTSNode() for _ in range(self.num_envs)]
        model = self.agent._orig_mod if hasattr(self.agent, "_orig_mod") else self.agent
        
        # Temporary state buffers for all envs
        temp_states = [create_string_buffer(self.c_struct_size) for _ in range(self.num_envs)]

        for _ in range(num_simulations):
            nodes = []
            accumulated_rewards = np.zeros(self.num_envs)
            dones = [False] * self.num_envs
            depths = np.zeros(self.num_envs)
            
            # 1. Selection Phase for all envs
            for i in range(self.num_envs):
                self.lib.get_game_state(state_ptrs[i], temp_states[i])
                node = roots[i]
                while node.children and depths[i] < 3:
                    # PUCT selection
                    total_visits = sum(c.visit_count for c in node.children.values())
                    sqrt_total = math.sqrt(total_visits) if total_visits > 0 else 0
                    node = max(node.children.values(), key=lambda c: c.value() + self.c_puct * c.prior * (sqrt_total / (1 + c.visit_count)))
                    
                    # Step C state
                    reward = c_float(0.0)
                    done = c_bool(False)
                    self.lib.step_game(temp_states[i], int(node.action_id), byref(reward), byref(done))
                    accumulated_rewards[i] += reward.value
                    depths[i] += 1
                    if done.value:
                        dones[i] = True
                        break
                nodes.append(node)

            # 2. Batched Expansion & Evaluation (GPU Batch)
            obs_batch = []
            for i in range(self.num_envs):
                self.lib.get_observation(temp_states[i], self.obs_buffer)
                obs_batch.append(list(self.obs_buffer))
            
            obs_tensor = torch.Tensor(obs_batch).to(self.device)
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    hidden = model._get_hidden(obs_tensor)
                    values = model.critic(hidden).flatten().cpu().numpy()
                    all_logits = model.actor(hidden)
            
            # Final values = Critic + rewards found during search
            final_values = values + (accumulated_rewards / 100.0)

            # Expand nodes that are not terminal
            mask_batch = []
            expand_indices = []
            for i in range(self.num_envs):
                if not dones[i] and depths[i] < 3:
                    self.lib.get_action_mask(temp_states[i], self.mask_buffer)
                    mask_batch.append(list(self.mask_buffer))
                    expand_indices.append(i)
                else:
                    mask_batch.append([0]*192) # Dummy mask
            
            masks_tensor = torch.Tensor(mask_batch).to(self.device)
            # Apply masks to logits to get priors
            priors_batch = all_logits + (masks_tensor == 0) * -1e9
            priors_batch = torch.softmax(priors_batch, dim=1).cpu().numpy()

            for i in expand_indices:
                node = nodes[i]
                mask_np = masks_tensor[i].cpu().numpy()
                legal_indices = np.where(mask_np > 0)[0]
                for idx in legal_indices:
                    if idx not in node.children:
                        node.children[idx] = MCTSNode(action_id=idx, parent=node, prior=priors_batch[i][idx])

            # 3. Backpropagation Phase
            for i in range(self.num_envs):
                node = nodes[i]
                val = final_values[i]
                while node:
                    node.visit_count += 1
                    node.value_sum += val
                    node = node.parent

        # Final actions and target distributions
        best_actions = []
        target_distributions = []
        for i in range(self.num_envs):
            # Target distribution is the visit count normalized
            dist = np.zeros(192)
            for action_id, child in roots[i].children.items():
                dist[action_id] = child.visit_count
            
            if dist.sum() > 0:
                dist = dist / dist.sum()
                best_action = np.argmax(dist)
            else:
                best_action = 0 # Fallback
                
            best_actions.append(best_action)
            target_distributions.append(dist)

        return np.array(best_actions), np.array(target_distributions)
