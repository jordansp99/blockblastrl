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
Start the high-speed marathon training. The script automatically detects and utilizes **CUDA** or **MPS** if available:
```bash
python train.py
```
Monitor the progress in real-time via TensorBoard:
```bash
tensorboard --logdir=runs
```

### Watching the AI Play
Load a saved checkpoint and watch the AI in slow-motion (1 second per move):
```bash
python play.py checkpoints/STRATEGIC_MARATHON_.../iter_X.pt
```

## Performance & Architecture

- **CUDA Optimization:** The PufferLib backend is optimized for batch processing, allowing NVIDIA GPUs to train on 128+ parallel games simultaneously without CPU bottlenecks.
- **Action Masking:** The AI is physically prevented from making invalid moves by the C engine, ensuring 100% of training data is spent on high-level strategy.
- **Global View Brain:** The CNN uses specialized 1x8 and 8x1 kernels to "see" entire rows and columns, mimicking human spatial reasoning.
