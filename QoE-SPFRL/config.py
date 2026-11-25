from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class TrainConfig:
    experiment_name: str = "qoe_spfrl_multimedia"
    scenario_name: str = "simple_CF_mMIMO"
    seed: int = 42
    device: str = "cpu"
    run_dir: str = "runs/qoe_spfrl00000"
    algorithm_name: str = "qoe_spfrl"

    episodes: int = 1200
    max_steps: int = 50
    batch_size: int = 128
    seq_len: int = 8
    buffer_size: int = 200000
    learning_starts: int = 200
    train_every: int = 8
    eval_every: int = 20
    save_every: int = 50
    log_every: int = 1
    eval_episodes: int = 5
    smooth_window: int = 50
    grad_clip_norm: float = 1.0

    gamma: float = 0.98
    lr: float = 1e-4
    mu: float = 0.5
    K: int = 2
    beta: float = 0.2
    sync_tau: float = 0.03
    p_fedavg_interval: int = 10
    reg_warmup_episodes: int = 100

    target_update_every: int = 40
    target_update_tau: float = 0.02
    cache_update_interval: int = 5
    preference_update_interval: int = 100000
    num_content_F: int = 100
    cache_curriculum_episodes: int = 80
    lamda: float = 0.2
    use_lstm: bool = True

    reward_scale: float = 1.0
    target_clip: float = 60.0
    q_reduce: str = "mean"
    max_grad_norm: float = 1.0

    epsilon_start: float = 1.0
    epsilon_decay: float = 0.98
    epsilon_min: float = 0.005
    cache_hit_reward: float = 320.0
    cache_miss_penalty: float = 80.0
    qoe_penalty_weight: float = 220.0
    edge_cache_delay_gain: float = 0.30
    edge_cache_energy_gain: float = 0.25
    cache_popularity_weight: float = 1.0
    cache_explore_pop_power: float = 0.7
    advantage_warmup_episodes: int = 100
    preference_noise: float = 0.01
    preserve_cache_on_reset: bool = True
    bs_cache_capacity: int = 40
    bs_compute_capacity: float = 50e9
    cache_policy_blend: float = 0.95
    cache_request_window: int = 1000
    cache_explore_epsilon_scale: float = 0.05
    use_qoe_action_prior: bool = True
    qoe_action_prior_explore_scale: float = 0.25
    method_name: str = "QoE-SPFRL"
    time_slots_per_day: int = 24
    demand_event_probability: float = 0.04
    demand_event_duration: int = 6
    demand_event_strength: float = 0.35
    spatial_demand_correlation: float = 0.05
    regional_demand_strength: float = 2.0
    resource_heterogeneity_strength: float = 1.8
    service_temporal_weight: float = 0.40
    region_q_t: tuple = (0.60, 0.75, 0.85, 0.80, 0.55)
    region_q_e: tuple = (0.40, 0.25, 0.15, 0.20, 0.45)
    qoe_reward_weight: float = 260.0
    delay_penalty_weight: float = 120.0
    energy_penalty_weight: float = 20.0
    overtime_penalty_weight: float = 180.0
    qoe_delay_slope: float = 1.4
    edge_collab_bonus: float = 40.0
    log_cache_interval: int = 50

    def to_namespace(self):
        return argparse.Namespace(**asdict(self))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training entry for QoE-SPFRL")
    parser.add_argument("--experiment-name", type=str, default=TrainConfig.experiment_name)
    parser.add_argument("--scenario-name", type=str, default=TrainConfig.scenario_name)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", type=str, default=TrainConfig.device)
    parser.add_argument("--run-dir", type=str, default=TrainConfig.run_dir)
    parser.add_argument("--algorithm-name", type=str, choices=["qoe_spfrl"], default=TrainConfig.algorithm_name)

    parser.add_argument("--episodes", type=int, default=TrainConfig.episodes)
    parser.add_argument("--max-steps", type=int, default=TrainConfig.max_steps)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--seq-len", type=int, default=TrainConfig.seq_len)
    parser.add_argument("--buffer-size", type=int, default=TrainConfig.buffer_size)
    parser.add_argument("--learning-starts", type=int, default=TrainConfig.learning_starts)
    parser.add_argument("--train-every", type=int, default=TrainConfig.train_every)
    parser.add_argument("--eval-every", type=int, default=TrainConfig.eval_every)
    parser.add_argument("--save-every", type=int, default=TrainConfig.save_every)
    parser.add_argument("--log-every", type=int, default=TrainConfig.log_every)
    parser.add_argument("--eval-episodes", type=int, default=TrainConfig.eval_episodes)
    parser.add_argument("--smooth-window", type=int, default=TrainConfig.smooth_window)
    parser.add_argument("--target-update-every", type=int, default=TrainConfig.target_update_every)
    parser.add_argument("--grad-clip-norm", type=float, default=TrainConfig.grad_clip_norm)

    parser.add_argument("--gamma", type=float, default=TrainConfig.gamma)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--mu", type=float, default=TrainConfig.mu)
    parser.add_argument("--K", type=int, default=TrainConfig.K)
    parser.add_argument("--beta", type=float, default=TrainConfig.beta)
    parser.add_argument("--sync-tau", type=float, default=TrainConfig.sync_tau)
    parser.add_argument("--p-fedavg-interval", type=int, default=TrainConfig.p_fedavg_interval)
    parser.add_argument("--reg-warmup-episodes", type=int, default=TrainConfig.reg_warmup_episodes)

    parser.add_argument("--epsilon-start", type=float, default=TrainConfig.epsilon_start)
    parser.add_argument("--epsilon-decay", type=float, default=TrainConfig.epsilon_decay)
    parser.add_argument("--epsilon-min", type=float, default=TrainConfig.epsilon_min)

    parser.add_argument("--target-update-tau", type=float, default=TrainConfig.target_update_tau)
    parser.add_argument("--cache-update-interval", type=int, default=TrainConfig.cache_update_interval)
    parser.add_argument("--preference-update-interval", type=int, default=TrainConfig.preference_update_interval)
    parser.add_argument("--num-content-F", type=int, default=TrainConfig.num_content_F)
    parser.add_argument("--cache-curriculum-episodes", type=int, default=TrainConfig.cache_curriculum_episodes)
    parser.add_argument("--cache-hit-reward", type=float, default=TrainConfig.cache_hit_reward)
    parser.add_argument("--cache-miss-penalty", type=float, default=TrainConfig.cache_miss_penalty)
    parser.add_argument("--qoe-penalty-weight", type=float, default=TrainConfig.qoe_penalty_weight)
    parser.add_argument("--edge-cache-delay-gain", type=float, default=TrainConfig.edge_cache_delay_gain)
    parser.add_argument("--edge-cache-energy-gain", type=float, default=TrainConfig.edge_cache_energy_gain)
    parser.add_argument("--cache-popularity-weight", type=float, default=TrainConfig.cache_popularity_weight)
    parser.add_argument("--cache-explore-pop-power", type=float, default=TrainConfig.cache_explore_pop_power)
    parser.add_argument("--advantage-warmup-episodes", type=int, default=TrainConfig.advantage_warmup_episodes)
    parser.add_argument("--preference-noise", type=float, default=TrainConfig.preference_noise)
    parser.add_argument("--bs-cache-capacity", type=int, default=TrainConfig.bs_cache_capacity)
    parser.add_argument("--bs-compute-capacity", type=float, default=TrainConfig.bs_compute_capacity)
    parser.add_argument("--cache-policy-blend", type=float, default=TrainConfig.cache_policy_blend)
    parser.add_argument("--cache-request-window", type=int, default=TrainConfig.cache_request_window)
    parser.add_argument("--cache-explore-epsilon-scale", type=float, default=TrainConfig.cache_explore_epsilon_scale)
    parser.add_argument("--qoe-action-prior-explore-scale", type=float, default=TrainConfig.qoe_action_prior_explore_scale)
    parser.add_argument("--method-name", type=str, default=TrainConfig.method_name)
    parser.add_argument("--time-slots-per-day", type=int, default=TrainConfig.time_slots_per_day)
    parser.add_argument("--demand-event-probability", type=float, default=TrainConfig.demand_event_probability)
    parser.add_argument("--demand-event-duration", type=int, default=TrainConfig.demand_event_duration)
    parser.add_argument("--demand-event-strength", type=float, default=TrainConfig.demand_event_strength)
    parser.add_argument("--spatial-demand-correlation", type=float, default=TrainConfig.spatial_demand_correlation)
    parser.add_argument("--regional-demand-strength", type=float, default=TrainConfig.regional_demand_strength)
    parser.add_argument("--resource-heterogeneity-strength", type=float, default=TrainConfig.resource_heterogeneity_strength)
    parser.add_argument("--service-temporal-weight", type=float, default=TrainConfig.service_temporal_weight)
    parser.add_argument("--region-q-t", type=float, nargs=5, default=TrainConfig.region_q_t)
    parser.add_argument("--region-q-e", type=float, nargs=5, default=TrainConfig.region_q_e)
    parser.add_argument("--qoe-reward-weight", type=float, default=TrainConfig.qoe_reward_weight)
    parser.add_argument("--delay-penalty-weight", type=float, default=TrainConfig.delay_penalty_weight)
    parser.add_argument("--energy-penalty-weight", type=float, default=TrainConfig.energy_penalty_weight)
    parser.add_argument("--overtime-penalty-weight", type=float, default=TrainConfig.overtime_penalty_weight)
    parser.add_argument("--qoe-delay-slope", type=float, default=TrainConfig.qoe_delay_slope)
    parser.add_argument("--edge-collab-bonus", type=float, default=TrainConfig.edge_collab_bonus)
    parser.add_argument("--log-cache-interval", type=int, default=TrainConfig.log_cache_interval)
    parser.add_argument(
        "--preserve-cache-on-reset",
        dest="preserve_cache_on_reset",
        action="store_true",
        default=TrainConfig.preserve_cache_on_reset,
    )
    parser.add_argument(
        "--no-preserve-cache-on-reset",
        dest="preserve_cache_on_reset",
        action="store_false",
    )
    parser.add_argument(
        "--use-qoe-action-prior",
        dest="use_qoe_action_prior",
        action="store_true",
        default=TrainConfig.use_qoe_action_prior,
    )
    parser.add_argument(
        "--no-qoe-action-prior",
        dest="use_qoe_action_prior",
        action="store_false",
    )
    return parser


def parse_config() -> TrainConfig:
    args = build_parser().parse_args()
    return TrainConfig(**vars(args))
