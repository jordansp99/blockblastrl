import torch
import numpy as np
import math
from ctypes import *

# Define the CNode struct to match blockblast_lib.c
class CNode(Structure):
    _fields_ = [
        ("action_id", c_int),
        ("prior", c_float),
        ("visit_count", c_int),
        ("value_sum", c_float)
    ]

class MCTSEngine:
    """Uses the same logic but for single play"""
    def __init__(self, agent, device, lib, c_struct_size):
        self.agent = agent
        self.device = device
        self.lib = lib
        self.c_struct_size = c_struct_size

    def search(self, root_state_ptr, num_simulations=100):
        # We can use the Batched engine with num_envs=1 for simplicity and speed
        batch_engine = BatchedMCTSEngine(self.agent, self.device, self.lib, self.c_struct_size, 1)
        actions, _ = batch_engine.search([root_state_ptr], num_simulations)
        return actions[0]

class BatchedMCTSEngine:
    """C-ACCELERATED VECTORIZED MCTS: THE TURBO ENGINE"""
    def __init__(self, agent, device, lib, c_struct_size, num_envs):
        self.agent = agent
        self.device = device
        self.lib = lib
        self.num_envs = num_envs
        self.c_struct_size = c_struct_size
        self.c_puct = 1.4
        
        # TREE STORAGE IN C-MEMORY (Flat arrays are much faster)
        # Each environment gets its own pool of nodes (max 192 legal moves)
        self.nodes_per_env = 192
        self.tree_data = (CNode * (num_envs * self.nodes_per_env))()
        self.num_children = (c_int * num_envs)()
        
        # Contiguous buffers for batch C-calls
        self.obs_buffer = np.zeros(num_envs * 139, dtype=np.int32)
        self.mask_buffer = np.zeros(num_envs * 192, dtype=np.int32)
        self.temp_states = [create_string_buffer(c_struct_size) for _ in range(num_envs)]
        self.temp_state_ptrs = (c_void_p * num_envs)(*[cast(s, c_void_p) for s in self.temp_states])
        
        # Pointers to individual node pools for C-logic
        self.node_pool_ptrs = (POINTER(CNode) * num_envs)()
        for i in range(num_envs):
            self.node_pool_ptrs[i] = cast(addressof(self.tree_data) + (i * self.nodes_per_env * sizeof(CNode)), POINTER(CNode))

    def search(self, root_ptrs, num_simulations=50):
        model = self.agent._orig_mod if hasattr(self.agent, "_orig_mod") else self.agent
        c_root_ptrs = (c_void_p * self.num_envs)(*root_ptrs)
        
        # 1. INITIAL EVALUATION (To get priors for the root)
        self.lib.get_observation_batch(c_root_ptrs, self.obs_buffer.ctypes.data_as(POINTER(c_int)), self.num_envs)
        self.lib.get_action_mask_batch(c_root_ptrs, self.mask_buffer.ctypes.data_as(POINTER(c_int)), self.num_envs)
        
        obs_tensor = torch.from_numpy(self.obs_buffer).view(self.num_envs, 139).float().to(self.device)
        masks_tensor = torch.from_numpy(self.mask_buffer).view(self.num_envs, 192).to(self.device)
        
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", enabled=(self.device.type == "cuda")):
                hidden, _ = model._get_hidden(obs_tensor)
                all_logits = model.actor(hidden)
                priors = torch.softmax(all_logits + (masks_tensor == 0) * -1e9, dim=1).cpu().numpy()

        # 2. INITIALIZE ROOT NODES IN C-MEMORY
        memset(addressof(self.tree_data), 0, sizeof(self.tree_data))
        for i in range(self.num_envs):
            mask_np = self.mask_buffer.reshape(self.num_envs, 192)[i]
            legal_indices = np.where(mask_np > 0)[0]
            self.num_children[i] = len(legal_indices)
            for j, idx in enumerate(legal_indices):
                self.tree_data[i * self.nodes_per_env + j].action_id = int(idx)
                self.tree_data[i * self.nodes_per_env + j].prior = float(priors[i][idx])

        # 3. MAIN SIMULATION LOOP (Selection and Backprop now in C)
        selected_actions = (c_int * self.num_envs)()
        
        for _ in range(num_simulations):
            # A. SELECTION (C-Accelerated + Game Step)
            self.lib.copy_game_state_batch(c_root_ptrs, self.temp_state_ptrs, self.num_envs)
            self.lib.mcts_select_batch(
                self.temp_state_ptrs, 
                cast(self.node_pool_ptrs, POINTER(c_void_p)), 
                self.num_children, 
                selected_actions, 
                self.num_envs, 
                self.c_puct
            )
            
            # B. EVALUATION (GPU Batch)
            self.lib.get_observation_batch(self.temp_state_ptrs, self.obs_buffer.ctypes.data_as(POINTER(c_int)), self.num_envs)
            obs_tensor = torch.from_numpy(self.obs_buffer).view(self.num_envs, 139).float().to(self.device)
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", enabled=(self.device.type == "cuda")):
                    hidden, _ = model._get_hidden(obs_tensor)
                    values = model.critic(hidden).flatten().cpu().numpy()
            
            # C. BACKPROPAGATION (C-Accelerated)
            child_indices = (c_int * self.num_envs)()
            for i in range(self.num_envs):
                child_indices[i] = -1
                for j in range(self.num_children[i]):
                    if self.tree_data[i * self.nodes_per_env + j].action_id == selected_actions[i]:
                        child_indices[i] = j
                        break
            
            c_values = (c_float * self.num_envs)(*values)
            self.lib.mcts_backprop_batch(
                cast(self.node_pool_ptrs, POINTER(c_void_p)), 
                child_indices, 
                c_values, 
                self.num_envs
            )

        # 4. EXTRACT RESULTS
        best_actions = np.zeros(self.num_envs, dtype=np.int32)
        target_dists = np.zeros((self.num_envs, 192), dtype=np.float32)
        for i in range(self.num_envs):
            dist = np.zeros(192)
            total_v = 0
            for j in range(self.num_children[i]):
                node = self.tree_data[i * self.nodes_per_env + j]
                dist[node.action_id] = node.visit_count
                total_v += node.visit_count
            
            if total_v > 0:
                dist /= total_v
                best_actions[i] = np.argmax(dist)
            target_dists[i] = dist

        return best_actions, target_dists
