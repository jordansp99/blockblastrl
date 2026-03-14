import gymnasium
import numpy as np
from gymnasium import spaces
from ctypes import *
import os

class BlockBlastEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, seed=None):
        self.render_mode = render_mode
        # Cross-platform library loading (macOS, Linux, Windows)
        dir_path = os.path.dirname(__file__)
        exts = [".dylib", ".so", ".dll"]
        lib_path = None
        for ext in exts:
            p = os.path.join(dir_path, f"libblockblast{ext}")
            if os.path.exists(p):
                lib_path = p
                break
            
        if not lib_path:
            raise FileNotFoundError(f"Could not find libblockblast with .dylib, .so, or .dll in {dir_path}. Please compile the C library first.")
            
        self.lib = cdll.LoadLibrary(lib_path)
        
        # Define C functions
        self.lib.init_game.argtypes = [c_int]
        self.lib.init_game.restype = c_void_p
        self.lib.reset_game.argtypes = [c_void_p]
        self.lib.get_observation.argtypes = [c_void_p, POINTER(c_int)]
        self.lib.get_action_mask.argtypes = [c_void_p, POINTER(c_int)]
        self.lib.step_game.argtypes = [c_void_p, c_int, POINTER(c_float), POINTER(c_bool)]
        self.lib.get_game_state.argtypes = [c_void_p, c_void_p]
        self.lib.set_game_state.argtypes = [c_void_p, c_void_p]
        
        # Batched functions
        self.lib.step_game_batch.argtypes = [POINTER(c_void_p), POINTER(c_int), POINTER(c_float), POINTER(c_bool), c_int]
        self.lib.get_observation_batch.argtypes = [POINTER(c_void_p), POINTER(c_int), c_int]
        self.lib.get_action_mask_batch.argtypes = [POINTER(c_void_p), POINTER(c_int), c_int]
        self.lib.copy_game_state_batch.argtypes = [POINTER(c_void_p), POINTER(c_void_p), c_int]
        
        # C-MCTS Batched
        self.lib.mcts_select_batch.argtypes = [POINTER(c_void_p), POINTER(c_void_p), POINTER(c_int), POINTER(c_int), c_int, c_float]
        self.lib.mcts_backprop_batch.argtypes = [POINTER(c_void_p), POINTER(c_int), POINTER(c_float), c_int]

        self.lib.render_game_state.argtypes = [c_void_p]
        self.lib.close_render.argtypes = []
        self.lib.free_game.argtypes = [c_void_p]

        self.state_ptr = self.lib.init_game(seed if seed is not None else -1)

        # Observation space: 
        # 64 for board, 75 for 3 shapes (3 * 25) -> 139 integers
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=0, high=1, shape=(139,), dtype=np.int32),
            "action_mask": spaces.Box(low=0, high=1, shape=(192,), dtype=np.int8)
        })
        self.action_space = spaces.Discrete(192)

        # Pre-allocate buffers for observation and mask
        self._obs_buffer = (c_int * 139)()
        self._mask_buffer = (c_int * 192)()

    def _get_obs(self):
        self.lib.get_observation(self.state_ptr, self._obs_buffer)
        obs_np = np.frombuffer(self._obs_buffer, dtype=np.int32).copy()
        
        self.lib.get_action_mask(self.state_ptr, self._mask_buffer)
        mask_np = np.frombuffer(self._mask_buffer, dtype=np.int32).astype(np.int8).copy()

        # Ensure at least one valid action to prevent crashing
        if mask_np.sum() == 0:
            mask_np[0] = 1
            
        return {"observation": obs_np, "action_mask": mask_np}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.lib.reset_game(self.state_ptr)
        return self._get_obs(), {}

    def step(self, action):
        reward = c_float(0.0)
        done = c_bool(False)
        self.lib.step_game(self.state_ptr, int(action), byref(reward), byref(done))
        
        if self.render_mode == "human":
            self.render()

        obs = self._get_obs()
        return obs, float(reward.value), bool(done.value), False, {}

    def render(self):
        if self.render_mode == "human":
            self.lib.render_game_state(self.state_ptr)

    def close(self):
        self.lib.close_render()
        self.lib.free_game(self.state_ptr)

gymnasium.register("BlockBlast-v0", entry_point=BlockBlastEnv)
