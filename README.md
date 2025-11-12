# Containerized MLOps Pipeline for Text Classification with DistilBERT

A reproducible NLP training workflow using PyTorch Lightning, Docker, and Weights & Biases.

## 📋 Overview

This project fine-tunes a **DistilBERT** model on the **MRPC** (Microsoft Research Paraphrase Corpus) task from the GLUE benchmark for paraphrase detection. It includes:

- **Structured training pipeline** with PyTorch Lightning
- **Experiment tracking** with Weights & Biases (wandb)
- **Model checkpointing** based on multiple metrics (F1, accuracy, validation loss)
- **Hyperparameter tuning** notebooks with Optuna
- **Docker support** for reproducible environments
- **Modular codebase** following MLOps best practices

## 🏗️ Project Structure

```
mlops/
├── main.py                      # Main entry point for training
├── pyproject.toml              # Python dependencies and project config
├── Dockerfile                  # Container definition for reproducible builds
├── src/
│   └── mlops_proj/
│       ├── data/
│       │   └── glue_data_module.py    # Data loading and preprocessing
│       ├── models/
│       │   └── glue_transformer.py     # Model definition with Lightning
│       └── training/
│           └── train.py                # Training logic
├── notebooks/
│   ├── mlops_hyperparameter_tuning.ipynb          # Manual HP tuning
│   └── mlops_hyperparameter_tuning_automatic.ipynb # Optuna-based tuning
├── models/
│   └── checkpoints/            # Saved model checkpoints
└── experiments/
    └── wandb/                  # W&B experiment logs
```

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip
- (Optional) NVIDIA GPU for faster training
- Weights & Biases account (free at [wandb.ai](https://wandb.ai))

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd mlops
   ```

2. **Install dependencies:**
   
   Using `uv` (recommended):
   ```bash
   pip install uv
   uv sync
   ```
   
   Or using `pip`:
   ```bash
   pip install -e .
   ```

3. **Set up Weights & Biases:**
   ```bash
   wandb login
   ```

4. **Create a `.env` file:**
   
   Create a `.env` file in the project root to configure environment variables:
   ```bash
   # .env
   WANDB_API_KEY=your_wandb_api_key_here
   WANDB_PROJECT=your_project_name
   ```
   
   **Important:** Add `.env` to your `.gitignore` to avoid committing sensitive credentials:
   ```bash
   echo ".env" >> .gitignore
   ```
   
   You can find your W&B API key at: https://wandb.ai/authorize

### Running Training

**Basic training with default parameters:**
```bash
python main.py
```

**Training with custom hyperparameters:**
```bash
python main.py \
  --learning_rate 5e-5 \
  --lr_schedule_type linear \
  --warmup_steps 100 \
  --weight_decay 0.01 \
  --train_batch_size 32 \
  --eval_batch_size 64 \
  --gradient_clip_val 1.0
```

### Available Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint_dir` | str | `models/checkpoints` | Directory to save model checkpoints |
| `--learning_rate` | float | `2e-5` | Learning rate for optimizer |
| `--lr_schedule_type` | str | `cosine` | Learning rate schedule (`linear`, `cosine`, or `constant`) |
| `--warmup_steps` | int | `50` | Number of warmup steps for LR scheduler |
| `--weight_decay` | float | `0.005` | Weight decay for AdamW optimizer |
| `--train_batch_size` | int | `16` | Batch size for training |
| `--eval_batch_size` | int | `32` | Batch size for evaluation |
| `--gradient_clip_val` | float | `0.5` | Gradient clipping value |

## 🐳 Docker Usage

The project includes a Dockerfile for reproducible training environments.

**Build the Docker image:**
```bash
docker build -t mlops-training .
```

**Run training in Docker:**

Before running, ensure your `.env` file is configured with your W&B credentials (see Installation step 4).

```bash
# With default parameters
docker run --rm -it --env-file .env \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/experiments:/app/experiments \
  mlops-training

# With custom parameters
docker run --rm -it --env-file .env \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/experiments:/app/experiments \
  mlops-training --learning_rate 5e-5 --lr_schedule_type linear
```

**Note:** The `--env-file .env` flag passes your environment variables (including W&B credentials) to the container.

## 📊 Experiment Tracking

All experiments are automatically logged to Weights & Biases, including:

- **Hyperparameters**: learning rate, batch sizes, warmup steps, etc.
- **Training metrics**: loss per step
- **Validation metrics**: accuracy, F1-score, validation loss
- **System metrics**: GPU usage, CPU, memory

Access your experiments at `https://wandb.ai/<your-username>/<project-name>`

## 📄 License

This project is provided for educational purposes.
