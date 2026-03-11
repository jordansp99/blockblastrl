# BlockBlast RL Lab

A high-performance Reinforcement Learning laboratory for solving the game BlockBlast. This project uses a hybrid architecture with a fast C game engine and a Python-based RL trainer, capable of reaching **18,000+ Steps Per Second (SPS)** on consumer hardware.

## Technical Stack

- **Game Engine:** Pure C (`blockblast_lib.c`) for maximum execution speed.
- **RL Framework:** [PufferLib](https://github.com/PufferAI/PufferLib) for high-speed vectorization and "zero-copy" data transfer.
- **Deep Learning:** PyTorch with a custom **Global View CNN** architecture (Spatial + Line-detector kernels).
- **Acceleration:** Native support for **NVIDIA (CUDA)** and **Apple Silicon (MPS)** GPUs.
- **Optimization:** Strategic Reward Shaping (Snugness, Hole Penalties, and Squared Line Bonuses).

## Installation

### 1. Prerequisites
- **macOS:** `brew install raylib`
- **Linux:** `sudo apt install libraylib-dev`

### 2. Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Compile C Library
```bash
# Detects OS (macOS/Linux) and builds .dylib or .so
make
```

## How to Use

### Training (Multi-GPU/CPU Agnostic)
Start high-speed marathon training. The script automatically detects and utilizes **CUDA** or **MPS** if available:
```bash
python train.py [ARGS]
```

**Common Arguments:**
- `--checkpoint PATH`: Load a saved checkpoint to resume training.
- `--run-name NAME`: Specify the run name for TensorBoard logs and checkpoint folder.
- `--start-update N`: Manually set the starting update number (useful for resetting legacy checkpoints to update 1).
- `--total-timesteps N`: Total training steps (default: 1,000,000,000).
- `--no-tensorboard`: Disable automatic TensorBoard launching.

**Example (Resuming a run):**
```bash
python train.py --checkpoint checkpoints/MY_RUN/update_150.pt --run-name MY_RUN
```

Monitor progress in real-time via TensorBoard (automatically launched by `train.py`):
```bash
tensorboard --logdir=runs
```

### Watching the AI Play
Load a saved checkpoint and watch the AI play in a slow-motion GUI:
```bash
python play.py [CHECKPOINT_PATH] [SEED] [--stochastic]
```

**Common Arguments:**
- `SEED`: Optional integer to fix the block sequence.
- `--stochastic`: By default, the AI plays **deterministically** (choosing the absolute best move). Use this flag to enable probabilistic sampling (how the AI behaves during training).

**Example:**
```bash
python play.py checkpoints/MY_RUN/update_150.pt 42
```
*Specifying a seed allows you to replay the same block sequence.*

## Performance & Architecture

- **CUDA Optimization:** The PufferLib backend is optimized for batch processing, allowing NVIDIA GPUs to train on 128+ parallel games simultaneously without CPU bottlenecks.
- **Action Masking:** The AI is physically prevented from making invalid moves by the C engine, ensuring 100% of training data is spent on high-level strategy.
- **Global View Brain:** The CNN uses specialized 1x8 and 8x1 kernels to "see" entire rows and columns, mimicking human spatial reasoning.
