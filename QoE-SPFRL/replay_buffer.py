from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import random
import numpy as np


@dataclass
class AgentSequenceTransition:
    share_obs_seq: np.ndarray
    actions_seq: np.ndarray
    rewards_seq: np.ndarray
    next_share_obs_seq: np.ndarray
    dones_seq: np.ndarray


class SequenceReplayBuffer:
    def reset(self):
        self.buffer.clear()

    def __init__(self, capacity: int, seq_len: int):
        self.capacity = capacity
        self.seq_len = seq_len
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, transition: AgentSequenceTransition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)

        share_obs = np.stack([t.share_obs_seq for t in batch], axis=0).astype(np.float32)
        actions = np.stack([t.actions_seq for t in batch], axis=0).astype(np.int64)
        rewards = np.stack([t.rewards_seq for t in batch], axis=0).astype(np.float32)
        next_share_obs = np.stack([t.next_share_obs_seq for t in batch], axis=0).astype(np.float32)
        dones = np.stack([t.dones_seq for t in batch], axis=0).astype(np.float32)

        return {
            "share_obs": share_obs,
            "actions": actions,
            "rewards": rewards,
            "next_share_obs": next_share_obs,
            "dones": dones,
        }