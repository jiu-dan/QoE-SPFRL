# QoE-SPFRL
# QoE-SPFRL

本仓库提供 QoE-SPFRL 的训练与评估代码，用于研究面向用户体验质量（QoE）的多媒体服务缓存、任务卸载与资源分配。

# 环境与依赖安装
使用 Python 3.10 或更高版本。请在仓库根目录执行以下命令。

# 1. 获取代码

```bash
git clone https://github.com/<用户名>/QoE-SPFRL.git
cd QoE-SPFRL
```

### 2. 创建虚拟环境

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux / macOS：

```bash
source .venv/bin/activate
```

### 3. 安装依赖

```bash
python -m pip install --upgrade pip
python -m pip install numpy torch gym pandas scikit-learn matplotlib tensorboard absl-py
```

主要依赖包括 PyTorch、NumPy、Gym、Pandas、scikit-learn、
Matplotlib、TensorBoard 和 absl-py。

## 运行方法

### 1. 快速检查

运行一个短回合，检查训练入口、日志记录和模型保存是否正常：

```bash
python main.py --device cpu --episodes 1 --max-steps 10 --save-every 1 --eval-every 1 --eval-episodes 1 --run-dir runs/smoke_test
```

该命令仅用于检查程序能否运行，不用于评估算法性能。

### 2. 模型训练

使用当前代码的默认训练参数，运行 1200 个回合：

```bash
python main.py --algorithm-name qoe_spfrl --device cpu --seed 42 --episodes 1200 --run-dir runs/qoe_spfrl_seed42
```

查看可配置参数：

```bash
python main.py --help
```

主要参数：

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `--episodes` | 训练回合数 | 1200 |
| `--max-steps` | 每回合最大步数 | 50 |
| `--seed` | 随机种子 | 42 |
| `--device` | 计算设备 | cpu |
| `--batch-size` | 训练批次大小 | 128 |
| `--eval-every` | 训练期间的评估间隔 | 20 |
| `--save-every` | 模型保存间隔 | 50 |
| `--run-dir` | 结果保存目录 | runs/qoe_spfrl00000 |

每次实验建议指定独立的 `--run-dir`，避免覆盖已有日志。

### 3. 模型评估

训练结束后，评估指定检查点：

```bash
python evaluate.py runs/qoe_spfrl_seed42/checkpoints/episode_1200.pt
```

评估脚本会读取该实验目录中的 `config.json`，运行 5 个评估回合，
并在终端输出平均回报、QoE、缓存命中率、时延、能耗等指标。

请保留检查点所在实验目录的结构，以及对应的 `config.json`。

### 4. 查看训练曲线

```bash
tensorboard --logdir runs/qoe_spfrl_seed42/tensorboard
```

打开终端提示的地址查看训练曲线。

## 输出文件

实验结果保存在 `--run-dir` 指定的目录中，主要包括：

- `config.json`：本次实验使用的参数。
- `metrics.csv`：训练指标。
- `eval_metrics.csv`：训练期间的评估指标。
- `checkpoints/`：模型检查点。
- `best_checkpoint.txt`：训练期间评估 QoE 最高的检查点路径。
- `tensorboard/`：TensorBoard 日志。

## 实验复现

当前默认配置见 `config.py`。复现论文实验时，应使用论文对应的参数、
随机种子及依赖版本，并在多个随机种子下重复实验，统计结果的均值和标准差。
