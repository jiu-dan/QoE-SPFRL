from __future__ import annotations

import json
import sys
from pathlib import Path

from config import TrainConfig
from train import evaluate_checkpoint


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python evaluate.py <checkpoint_path>")
    checkpoint_path = sys.argv[1]

    ckpt_path = Path(checkpoint_path)
    config_path = ckpt_path.parent.parent / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json for checkpoint: {config_path}")

    cfg = TrainConfig(**json.loads(config_path.read_text(encoding="utf-8")))
    metrics = evaluate_checkpoint(cfg, checkpoint_path, episodes=5)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
