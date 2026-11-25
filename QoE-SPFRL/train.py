import json
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict
from collections import deque

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from config import TrainConfig
from offpolicy.algorithms.mdqn.algorithm.pFedMe_DQN import QoE_SPFRL
from offpolicy.envs.mpe.MPE_Env import MPEEnv
from logger import CSVLogger
from replay_buffer import SequenceReplayBuffer, AgentSequenceTransition


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_algorithm_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace("+", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )


def resolve_algorithm_class(name: str):
    normalized = normalize_algorithm_name(name)

    if normalized in {"qoe_spfrl", "qoe_spfrl_multimedia", "spfrl", "pfedme", "pfedme_dqn"}:
        return QoE_SPFRL
    raise ValueError(
        f"Unknown algorithm_name={name!r}. "
        "Only qoe_spfrl is supported."
    )


def build_env_and_agent(cfg: TrainConfig):
    env_args = cfg.to_namespace()
    env = MPEEnv(env_args)
    env.seed(cfg.seed)

    obs_n = env.reset()
    obs = np.asarray(obs_n, dtype=np.float32)
    share_obs = obs.reshape(-1).astype(np.float32)

    action_dims = list(env.action_space[0].nvec.tolist())
    env_args.action_dims = action_dims
    env_args.action_dim = sum(action_dims)
    env_args.bs_positions = [tuple(agent.location) for agent in env.world.agents]

    env_args.num_content_F = env.world.num_content_F
    env_args.num_users = env.world.num_users
    env_args.cache_capacities = [int(agent.max_content_cached) for agent in env.world.agents]

    cent_obs_dim = int(env.share_observation_space[0].shape[0])
    agent_cls = resolve_algorithm_class(getattr(cfg, "algorithm_name", "qoe_spfrl"))
    agent = agent_cls(
        args=env_args,
        num_agents=env.num_agents,
        cent_obs_dim=cent_obs_dim,
        action_dim=sum(action_dims),
        device=torch.device(cfg.device),
        mu=cfg.mu,
        lr=cfg.lr,
        K=cfg.K,
    )
    agent.world = env.world

    return env, agent, action_dims, obs, share_obs


def build_padded_state_seq(state_history, seq_len: int):
    hist = list(state_history)
    if len(hist) == 0:
        raise ValueError("state_history is empty")
    while len(hist) < seq_len:
        hist.insert(0, hist[0])
    hist = hist[-seq_len:]
    return np.stack(hist, axis=0).astype(np.float32)


def select_actions(agent, share_obs_seq: np.ndarray, action_dims, epsilon: float, num_agents: int):
    actions = []
    state_seq = np.expand_dims(share_obs_seq, axis=0)
    for agent_idx in range(num_agents):
        action_vec = agent.select_action(agent_idx, state_seq, epsilon=epsilon)[0]
        actions.append(action_vec)
    return np.asarray(actions, dtype=np.int64)


def save_checkpoint(agent, run_dir: Path, episode: int) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"episode_{episode:04d}.pt"
    state_dict = (
        agent.checkpoint_state_dict()
        if hasattr(agent, "checkpoint_state_dict")
        else agent.state_dict()
    )
    torch.save(state_dict, ckpt_path)
    return ckpt_path


def mean_recent(values, window: int):
    if not values:
        return np.nan
    return float(np.mean(values[-window:]))


def run_training(cfg: TrainConfig) -> Dict:
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.json")
    logger = CSVLogger(str(run_dir / "metrics.csv"), append=False)
    eval_logger = CSVLogger(str(run_dir / "eval_metrics.csv"), append=False)
    print(f"[Run] algorithm={getattr(cfg, 'algorithm_name', 'qoe_spfrl')} run_dir={run_dir}")

    tb_dir = run_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))

    set_seed(cfg.seed)
    env, agent, action_dims, obs, share_obs = build_env_and_agent(cfg)
    env.world.debug_reward = True
    replay_buffers = [SequenceReplayBuffer(cfg.buffer_size, cfg.seq_len) for _ in range(env.num_agents)]

    for buf in replay_buffers:
        buf.reset()

    global_step = 0
    epsilon = cfg.epsilon_start
    best_eval_qoe = -float("inf")
    summary = {}

    reward_return_history = []
    hit_history = []
    qoe_history = []
    loss_history = []
    overtime_history = []

    for episode in range(1, cfg.episodes + 1):
        if hasattr(agent, 'set_current_episode'):
            agent.set_current_episode(episode)

        obs_n = env.reset()
        obs = np.asarray(obs_n, dtype=np.float32)
        share_obs = obs.reshape(-1).astype(np.float32)

        episode_reward = 0.0
        episode_delay = 0.0
        episode_energy = 0.0
        episode_hit = 0.0
        episode_qoe = 0.0
        episode_overtime = 0.0
        episode_edge_hit = 0.0
        episode_neighbor_hit = 0.0
        episode_loss = 0.0
        episode_updates = 0
        actual_steps = 0

        time_action = 0.0
        time_env = 0.0
        time_buffer = 0.0
        time_train = 0.0

        state_history = deque(maxlen=cfg.seq_len)
        state_history.append(share_obs.copy())
        episode_steps = []

        for _ in range(cfg.max_steps):
            share_obs_seq = build_padded_state_seq(state_history, cfg.seq_len)

            t0 = time.perf_counter()
            actions = select_actions(agent, share_obs_seq, action_dims, epsilon, env.num_agents)
            time_action += time.perf_counter() - t0

            t0 = time.perf_counter()
            next_obs_n, reward_n, done_n, info_n, delay_n, energy_n, hit_n, qoe_n = env.step(actions)
            time_env += time.perf_counter() - t0

            next_obs = np.asarray(next_obs_n, dtype=np.float32)
            next_share_obs = next_obs.reshape(-1).astype(np.float32)
            rewards = np.asarray(reward_n, dtype=np.float32).reshape(env.num_agents, 1)
            dones = np.asarray(done_n, dtype=np.float32).reshape(env.num_agents, 1)
            delays = np.asarray(delay_n, dtype=np.float32)
            energies = np.asarray(energy_n, dtype=np.float32)
            hits = np.asarray(hit_n, dtype=np.float32)
            qoes = np.asarray(qoe_n, dtype=np.float32)
            overtime_values = np.asarray(
                [info.get("overtime_ratio", 0.0) for info in info_n],
                dtype=np.float32,
            )
            edge_hit_values = np.asarray(
                [info.get("edge_hit_rate", 0.0) for info in info_n],
                dtype=np.float32,
            )
            neighbor_hit_values = np.asarray(
                [info.get("neighbor_hit_rate", 0.0) for info in info_n],
                dtype=np.float32,
            )

            t0 = time.perf_counter()
            episode_steps.append({
                "share_obs": share_obs.copy(),
                "actions": actions.copy(),
                "rewards": rewards.copy(),
                "next_share_obs": next_share_obs.copy(),
                "dones": dones.copy(),
            })

            if len(episode_steps) >= cfg.seq_len:
                seq_steps = episode_steps[-cfg.seq_len:]
                share_obs_seq_train = np.stack([s["share_obs"] for s in seq_steps], axis=0).astype(np.float32)
                actions_seq_train = np.stack([s["actions"] for s in seq_steps], axis=0).astype(np.int64)
                rewards_seq_train = np.stack([s["rewards"] for s in seq_steps], axis=0).astype(np.float32)
                next_share_obs_seq_train = np.stack([s["next_share_obs"] for s in seq_steps], axis=0).astype(np.float32)
                dones_seq_train = np.stack([s["dones"] for s in seq_steps], axis=0).astype(np.float32)

                for agent_idx in range(env.num_agents):
                    replay_buffers[agent_idx].push(AgentSequenceTransition(
                        share_obs_seq=share_obs_seq_train,
                        actions_seq=actions_seq_train[:, agent_idx, :],
                        rewards_seq=rewards_seq_train[:, agent_idx, :],
                        next_share_obs_seq=next_share_obs_seq_train,
                        dones_seq=dones_seq_train[:, agent_idx, :],
                    ))
            time_buffer += time.perf_counter() - t0

            obs = next_obs
            share_obs = next_share_obs
            state_history.append(share_obs.copy())

            global_step += 1
            actual_steps += 1

            episode_reward += float(rewards.mean())
            episode_delay += float(delays.mean())
            episode_energy += float(energies.mean())
            episode_hit += float(hits.mean())
            episode_qoe += float(qoes.mean())
            episode_overtime += float(overtime_values.mean())
            episode_edge_hit += float(edge_hit_values.mean())
            episode_neighbor_hit += float(neighbor_hit_values.mean())

            if global_step % cfg.train_every == 0:
                ready_agents = [
                    i for i in range(env.num_agents)
                    if len(replay_buffers[i]) >= max(cfg.learning_starts, cfg.batch_size)
                ]
                if len(ready_agents) > 0:
                    t0 = time.perf_counter()
                    batch_dict = {i: replay_buffers[i].sample(cfg.batch_size) for i in ready_agents}
                    train_info, _, _ = agent.train_policy_on_batch(batch_dict)
                    loss_value = float(train_info.get("average_loss", 0.0))
                    episode_loss += loss_value
                    episode_updates += 1

                    if hasattr(agent, "update_target_networks") and global_step % cfg.target_update_every == 0:
                        agent.update_target_networks()
                    time_train += time.perf_counter() - t0

            if bool(dones.all()):
                break

        epsilon = max(cfg.epsilon_min, epsilon * cfg.epsilon_decay)

        reward_return = episode_reward
        delay_mean = episode_delay / max(1, actual_steps)
        energy_mean = episode_energy / max(1, actual_steps)
        hit_mean = episode_hit / max(1, actual_steps)
        qoe_mean = episode_qoe / max(1, actual_steps)
        overtime_mean = episode_overtime / max(1, actual_steps)
        edge_hit_mean = episode_edge_hit / max(1, actual_steps)
        neighbor_hit_mean = episode_neighbor_hit / max(1, actual_steps)
        loss_mean = episode_loss / max(1, episode_updates) if episode_updates > 0 else 0.0
        cache_matrix = np.asarray([a.state.content_state for a in env.world.agents], dtype=np.float32)
        cache_diversity = float(np.mean(np.std(cache_matrix, axis=0))) if cache_matrix.size else 0.0
        cache_fill_ratio = float(cache_matrix.mean()) if cache_matrix.size else 0.0

        reward_return_history.append(reward_return)
        hit_history.append(hit_mean)
        qoe_history.append(qoe_mean)
        overtime_history.append(overtime_mean)
        if episode_updates > 0:
            loss_history.append(loss_mean)

        row = {
            "episode": episode,
            "global_step": global_step,
            "epsilon": epsilon,
            "episode_steps": actual_steps,
            "reward_return": reward_return,
            "reward_return_smooth": mean_recent(reward_return_history, cfg.smooth_window),
            "delay_mean": delay_mean,
            "energy_mean": energy_mean,
            "hit_mean": hit_mean,
            "hit_smooth": mean_recent(hit_history, cfg.smooth_window),
            "edge_hit_mean": edge_hit_mean,
            "neighbor_hit_mean": neighbor_hit_mean,
            "cache_diversity": cache_diversity,
            "cache_fill_ratio": cache_fill_ratio,
            "qoe_mean": qoe_mean,
            "qoe_smooth": mean_recent(qoe_history, cfg.smooth_window),
            "overtime_mean": overtime_mean,
            "overtime_smooth": mean_recent(overtime_history, cfg.smooth_window),
            "loss_mean": loss_mean,
            "loss_smooth": mean_recent(loss_history, cfg.smooth_window),
            "episode_updates": episode_updates,
            "time_action": time_action,
            "time_env": time_env,
            "time_buffer": time_buffer,
            "time_train": time_train,
            "time_total": time_action + time_env + time_buffer + time_train,
        }
        logger.log(row)
        cache_log_interval = int(getattr(cfg, "log_cache_interval", 50))
        if cache_log_interval > 0 and episode % cache_log_interval == 0:
            cache_matrix = np.array([agent.state.content_state for agent in env.world.agents])

            print(f"\n[Cache Observation at Episode {episode}]")
            print(f"Agent 0 (First 10): {cache_matrix[0, :10]}")
            print(f"Agent 1 (First 10): {cache_matrix[1, :10]}")

            mean_cache = cache_matrix.mean(axis=0)
            print(f"Cache Diversity Score (Std dev of mean across contents): {np.std(mean_cache):.4f}")

            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 4))
            plt.imshow(cache_matrix, aspect='auto', cmap='Greens', interpolation='nearest')
            plt.title(f"Agent Cache Heterogeneity at Episode {episode}")
            plt.xlabel("Content ID")
            plt.ylabel("Agent ID")
            plt.savefig(run_dir / f"cache_heatmap_ep{episode:04d}.png")
            plt.close()

        writer.add_scalar("Episode/Reward_return", row["reward_return"], episode)
        writer.add_scalar("Episode/Hit_mean", row["hit_mean"], episode)
        writer.add_scalar("Episode/Edge_hit_mean", row["edge_hit_mean"], episode)
        writer.add_scalar("Episode/Neighbor_hit_mean", row["neighbor_hit_mean"], episode)
        writer.add_scalar("Episode/Delay_mean", row["delay_mean"], episode)
        writer.add_scalar("Episode/Energy_mean", row["energy_mean"], episode)
        writer.add_scalar("Episode/QoE_mean", row["qoe_mean"], episode)
        writer.add_scalar("Episode/Overtime_mean", row["overtime_mean"], episode)
        writer.add_scalar("Episode/Cache_diversity", row["cache_diversity"], episode)
        writer.add_scalar("Episode/Loss_mean", row["loss_mean"], episode)
        writer.add_scalar("Episode/Epsilon", row["epsilon"], episode)

        if getattr(env.world, 'debug_reward', False):
            agent0 = env.world.agents[0]
            info = getattr(agent0, 'last_reward_info', None)
            if info is not None:
                print(
                    f"[Episode {episode:04d} summary] "
                    f"EpisodeReturn={row['reward_return']:.4f}, "
                    f"QoEMean={row['qoe_mean']:.4f} | "
                    f"Agent 0 final step: Reward={info['reward']:.4f}, "
                    f"QoE={info.get('qoe_score', 0.0):.4f}, "
                    f"AvgCost={info['avg_cost']:.4f}, "
                    f"HitRate={info['hit_rate']:.4f}, "
                    f"EdgeHit={info.get('edge_hit_rate', 0.0):.4f}, "
                    f"AvgDelay={info['avg_delay']:.6f}s, "
                    f"AvgEnergy={info['avg_energy']:.10f}J, "
                    f"OvertimeRatio={info['overtime_ratio']:.4f}"
                )

        if episode % cfg.save_every == 0:
            save_checkpoint(agent, run_dir, episode)

        if episode % cfg.eval_every == 0:
            eval_metrics = evaluate_isolated(agent, cfg, action_dims)
            eval_logger.log({"episode": episode, **eval_metrics})
            with (run_dir / f"eval_episode_{episode:04d}.json").open("w", encoding="utf-8") as f:
                json.dump(eval_metrics, f, indent=2, ensure_ascii=False)
            if eval_metrics["qoe_mean"] > best_eval_qoe:
                best_eval_qoe = eval_metrics["qoe_mean"]
                best_path = save_checkpoint(agent, run_dir, episode)
                (run_dir / "best_checkpoint.txt").write_text(str(best_path), encoding="utf-8")
            print(
                f"[Eval {episode:04d}] "
                f"reward_return={eval_metrics['reward_return_mean']:.6f} "
                f"hit={eval_metrics['hit_mean']:.4f} "
                f"edge_hit={eval_metrics['edge_hit_mean']:.4f} "
                f"  delay={eval_metrics['delay_mean']:.6f} "
                f"energy={eval_metrics['energy_mean']:.6f} "
                f"qoe={eval_metrics['qoe_mean']:.4f} "
                f"overtime={eval_metrics['overtime_mean']:.4f}"
            )

        summary = row

    writer.close()
    return summary


def evaluate(
    agent,
    env,
    action_dims,
    episodes: int = 5,
    seq_len: int = 4,
    max_steps: int | None = None,
) -> Dict:
    reward_returns = []
    delays = []
    energies = []
    hits = []
    qoes = []
    overtimes = []
    edge_hits = []
    neighbor_hits = []

    for _ in range(episodes):
        obs_n = env.reset()
        obs = np.asarray(obs_n, dtype=np.float32)
        share_obs = obs.reshape(-1).astype(np.float32)

        state_history = deque(maxlen=seq_len)
        state_history.append(share_obs.copy())

        ep_reward = ep_delay = ep_energy = ep_hit = ep_qoe = 0.0
        ep_overtime = ep_edge_hit = ep_neighbor_hit = 0.0
        ep_steps = 0

        episode_steps = env.world_length if max_steps is None else int(max_steps)
        for _ in range(episode_steps):
            share_obs_seq = build_padded_state_seq(state_history, seq_len)
            actions = select_actions(agent, share_obs_seq, action_dims, epsilon=0.0, num_agents=env.num_agents)

            next_obs_n, reward_n, done_n, info_n, delay_n, energy_n, hit_n, qoe_n = env.step(actions)

            obs = np.asarray(next_obs_n, dtype=np.float32)
            share_obs = obs.reshape(-1).astype(np.float32)
            state_history.append(share_obs.copy())

            ep_reward += float(np.asarray(reward_n).mean())
            ep_delay += float(np.asarray(delay_n).mean())
            ep_energy += float(np.asarray(energy_n).mean())
            ep_hit += float(np.asarray(hit_n).mean())
            ep_qoe += float(np.asarray(qoe_n).mean())
            ep_overtime += float(np.mean([info.get("overtime_ratio", 0.0) for info in info_n]))
            ep_edge_hit += float(np.mean([info.get("edge_hit_rate", 0.0) for info in info_n]))
            ep_neighbor_hit += float(np.mean([info.get("neighbor_hit_rate", 0.0) for info in info_n]))
            ep_steps += 1

            if bool(np.asarray(done_n).all()):
                break

        reward_returns.append(ep_reward)
        delays.append(ep_delay / max(1, ep_steps))
        energies.append(ep_energy / max(1, ep_steps))
        hits.append(ep_hit / max(1, ep_steps))
        qoes.append(ep_qoe / max(1, ep_steps))
        overtimes.append(ep_overtime / max(1, ep_steps))
        edge_hits.append(ep_edge_hit / max(1, ep_steps))
        neighbor_hits.append(ep_neighbor_hit / max(1, ep_steps))

    return {
        "episodes": episodes,
        "reward_return_mean": float(np.mean(reward_returns)),
        "reward_return_std": float(np.std(reward_returns)),
        "delay_mean": float(np.mean(delays)),
        "delay_std": float(np.std(delays)),
        "energy_mean": float(np.mean(energies)),
        "energy_std": float(np.std(energies)),
        "hit_mean": float(np.mean(hits)),
        "hit_std": float(np.std(hits)),
        "edge_hit_mean": float(np.mean(edge_hits)),
        "neighbor_hit_mean": float(np.mean(neighbor_hits)),
        "qoe_mean": float(np.mean(qoes)),
        "qoe_std": float(np.std(qoes)),
        "overtime_mean": float(np.mean(overtimes)),
        "overtime_std": float(np.std(overtimes)),
    }


def evaluate_isolated(agent, cfg: TrainConfig, action_dims) -> Dict:
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.random.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    training_world = getattr(agent, "world", None)

    try:
        eval_seed = int(cfg.seed) + 10_000
        set_seed(eval_seed)
        eval_env = MPEEnv(cfg.to_namespace())
        eval_env.seed(eval_seed)
        eval_env.world.debug_reward = False
        agent.world = eval_env.world
        metrics = evaluate(
            agent,
            eval_env,
            action_dims,
            episodes=cfg.eval_episodes,
            seq_len=cfg.seq_len,
            max_steps=cfg.max_steps,
        )
        metrics["seed"] = eval_seed
        return metrics
    finally:
        agent.world = training_world
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.random.set_rng_state(torch_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)


def evaluate_checkpoint(cfg: TrainConfig, checkpoint_path: str, episodes: int = 5) -> Dict:
    eval_seed = int(cfg.seed) + 10_000
    set_seed(eval_seed)
    eval_cfg = replace(cfg, seed=eval_seed)
    env, agent, action_dims, _, _ = build_env_and_agent(eval_cfg)
    state_dict = torch.load(checkpoint_path, map_location=cfg.device)
    if hasattr(agent, "load_checkpoint_state"):
        agent.load_checkpoint_state(state_dict)
    else:
        agent.load_state_dict(state_dict)
    if hasattr(agent, "set_current_episode"):
        try:
            checkpoint_episode = int(Path(checkpoint_path).stem.rsplit("_", 1)[-1])
        except ValueError:
            checkpoint_episode = cfg.episodes
        agent.set_current_episode(checkpoint_episode)
    metrics = evaluate(
        agent,
        env,
        action_dims,
        episodes=episodes,
        seq_len=cfg.seq_len,
        max_steps=cfg.max_steps,
    )
    metrics["seed"] = eval_seed
    return metrics
