# BlockBlast RL Lab

A high-performance Reinforcement Learning laboratory for solving the game BlockBlast. This project uses a hybrid architecture with a fast C game engine and a Python-based RL trainer.

## Technical Stack

- **Game Engine:** Pure C (`blockblast_lib.c`) for maximum execution speed.
- **RL Framework:** [PufferLib](https://github.com/PufferAI/PufferLib) for high-speed vectorization and "zero-copy" data transfer.
- **Deep Learning:** PyTorch with CNN architecture for spatial pattern recognition.
- **Optimization:** [Optuna](https://optuna.org/) for automated Bayesian hyperparameter sweeps and pruning.
- **Acceleration:** Support for Apple Metal (MPS) and NVIDIA (CUDA) GPUs.
- **GUI:** Raylib for real-time visualization of the AI's gameplay.

## Installation

### 1. Prerequisites
- **macOS:** `brew install raylib`
- **Linux (NVIDIA):** `sudo apt install libraylib-dev` (or build from source)

### 2. Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Compile C Library
```bash
make
```

## How to Use

### Training (Hyperparameter Sweep)
Start the automated Optuna sweep to find the best policy:
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
python play.py checkpoints/Trial_X_.../iter_976.pt cnn
```

## Architecture Details

- **Action Masking:** The AI is physically prevented from making invalid moves by the C engine, forcing 100% of training time to be spent on strategy.
- **Spatial Intelligence:** The CNN treats the 8x8 board as an image, allowing it to recognize rows, columns, and shape fitment patterns.
- **Persistent Studies:** Hyperparameter results are stored in `optuna_study.db` (SQLite), allowing the sweep to continue across restarts.
