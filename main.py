import argparse
from src.mlops_proj.training.train import train


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--checkpoint_dir", type=str, default="models/checkpoints")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lr_schedule_type", type=str, default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.005)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_clip_val", type=float, default=0.5)
    
    return parser.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()