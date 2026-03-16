import torch

class TorchBlockBlastEnv:
    def __init__(self, num_envs=128, device="cpu"):
        self.num_envs = num_envs
        self.device = device
        
        self.shapes_tensor = torch.zeros((17, 5, 5), dtype=torch.int32, device=device)
        shapes_data = [
            (1, 1, [[1]]),
            (2, 2, [[1,1], [1,1]]),
            (3, 3, [[1,1,1], [1,1,1], [1,1,1]]),
            (3, 1, [[1,1,1]]),
            (1, 3, [[1],[1],[1]]),
            (4, 1, [[1,1,1,1]]),
            (1, 4, [[1],[1],[1],[1]]),
            (5, 1, [[1,1,1,1,1]]),
            (1, 5, [[1],[1],[1],[1],[1]]),
            (2, 2, [[1,0], [1,1]]),
            (2, 2, [[0,1], [1,1]]),
            (2, 2, [[1,1], [1,0]]),
            (2, 2, [[1,1], [0,1]]),
            (3, 3, [[1,0,0], [1,0,0], [1,1,1]]),
            (3, 3, [[0,0,1], [0,0,1], [1,1,1]]),
            (3, 2, [[1,1,1], [0,1,0]]),
            (2, 3, [[1,0], [1,1], [1,0]])
        ]
        for i, (w, h, grid) in enumerate(shapes_data):
            for r in range(h):
                for c in range(w):
                    self.shapes_tensor[i, r, c] = grid[r][c]
                    
        self.shapes_tensor_float = self.shapes_tensor.float().unsqueeze(1)
        
        self.boards = torch.zeros((num_envs, 8, 8), dtype=torch.int32, device=device)
        self.current_shapes = torch.zeros((num_envs, 3), dtype=torch.int64, device=device)
        self.shape_active = torch.ones((num_envs, 3), dtype=torch.bool, device=device)
        self.dones = torch.zeros(num_envs, dtype=torch.bool, device=device)
        
        self.batch_indices = torch.arange(num_envs, device=device)
        
        dr = torch.arange(5, device=device).unsqueeze(1).expand(5, 5).flatten()
        dc = torch.arange(5, device=device).unsqueeze(0).expand(5, 5).flatten()
        self.offsets = dr * 12 + dc

    def generate_shapes(self, mask):
        num_needs = mask.sum().item()
        if num_needs > 0:
            self.current_shapes[mask] = torch.randint(0, 17, (num_needs, 3), device=self.device)
            self.shape_active[mask] = True

    def get_obs(self):
        boards_flat = self.boards.view(self.num_envs, 64)
        shapes_grids = self.shapes_tensor[self.current_shapes]
        active_mask = self.shape_active.view(self.num_envs, 3, 1, 1)
        shapes_grids = shapes_grids * active_mask
        shapes_flat = shapes_grids.view(self.num_envs, 75)
        return torch.cat([boards_flat, shapes_flat], dim=1)

    def get_action_mask(self):
        padded = torch.ones((self.num_envs, 1, 12, 12), dtype=torch.float32, device=self.device)
        padded[:, :, :8, :8] = self.boards.float().unsqueeze(1)
        
        overlaps = torch.nn.functional.conv2d(padded, self.shapes_tensor_float)
        can_place = (overlaps == 0)
        
        can_place_flat = can_place.view(self.num_envs, 17, 64)
        idx = self.current_shapes.unsqueeze(2).expand(-1, -1, 64)
        env_can_place = torch.gather(can_place_flat, 1, idx)
        
        active_mask = self.shape_active.unsqueeze(2)
        env_can_place = env_can_place & active_mask
        
        action_mask = env_can_place.view(self.num_envs, 192)
        has_valid = action_mask.any(dim=1)
        action_mask[~has_valid, 0] = True
        return action_mask

    def reset(self):
        self.boards.zero_()
        self.dones.zero_()
        self.generate_shapes(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        
        obs = self.get_obs()
        mask = self.get_action_mask()
        return torch.cat([mask.float(), obs.float()], dim=1)

    def step(self, action):
        shape_idx = action // 64
        pos = action % 64
        row = pos // 8
        col = pos % 8
        
        mask = self.get_action_mask()
        is_valid = mask[self.batch_indices, action] & ~self.dones
        
        base_idx = row * 12 + col
        indices = base_idx.unsqueeze(1) + self.offsets.unsqueeze(0)
        
        selected_shapes = self.current_shapes[self.batch_indices, shape_idx]
        shape_flat = self.shapes_tensor[selected_shapes].view(self.num_envs, 25)
        
        add_board = torch.zeros((self.num_envs, 144), dtype=torch.int32, device=self.device)
        add_board.scatter_(1, indices, shape_flat)
        add_board = add_board.view(self.num_envs, 12, 12)[:, :8, :8]
        
        self.boards = torch.where(is_valid.view(-1, 1, 1), self.boards + add_board, self.boards)
        
        active_update = self.shape_active.clone()
        active_update[self.batch_indices, shape_idx] = False
        self.shape_active = torch.where(is_valid.unsqueeze(1), active_update, self.shape_active)
        
        rows_sum = self.boards.sum(dim=2)
        cols_sum = self.boards.sum(dim=1)
        rows_full = (rows_sum == 8)
        cols_full = (cols_sum == 8)
        lines_cleared = rows_full.sum(dim=1) + cols_full.sum(dim=1)
        
        row_mask = ~rows_full.unsqueeze(2)
        col_mask = ~cols_full.unsqueeze(1)
        keep_mask = row_mask & col_mask
        self.boards = self.boards * keep_mask
        
        all_used = ~self.shape_active.any(dim=1)
        self.generate_shapes(all_used)
        
        new_mask = self.get_action_mask()
        has_moves = new_mask.any(dim=1)
        
        dones = ~has_moves | ~is_valid
        
        # Reward 10.0 for staying alive + 5.0 per line cleared, 0.0 if the move ends the game
        rewards = torch.where(dones, 0.0, 10.0 + lines_cleared.float() * 5.0)
        
        reset_mask = dones.clone()
        if reset_mask.any():
            self.boards[reset_mask] = 0
            self.generate_shapes(reset_mask)
            self.dones[reset_mask] = False
            
        next_obs = self.get_obs()
        next_mask = self.get_action_mask()
        
        self.dones = dones
        infos = {"lines_cleared": lines_cleared.cpu().numpy()}
        
        return torch.cat([next_mask.float(), next_obs.float()], dim=1), rewards, dones, dones, infos
