from config import parse_config
from train import run_training

if __name__ == "__main__":
    cfg = parse_config()
    print(f"算法: {cfg.algorithm_name}")
    print(f"结果目录: {cfg.run_dir}")
    print(f"渐进调度: {cfg.advantage_warmup_episodes} 回合")
    run_training(cfg)
