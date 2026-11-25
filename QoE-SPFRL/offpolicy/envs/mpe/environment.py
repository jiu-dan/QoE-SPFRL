import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from .multi_discrete import MultiDiscrete
from myUnits import get_adj, get_gain


class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, post_step_callback=None,
                 shared_viewer=True, discrete_action=True):

        self.world = world
        self.world_length = self.world.world_length
        self.agents = self.world.agents
        self.current_step = 0

        self.num_agents = len(world.agents)
        self.num_content_F = world.num_content_F

        self.dim_p = self.world.dim_p
        self.dim_w = self.world.dim_w

        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        self.post_step_callback = post_step_callback

        self.discrete_action_space = discrete_action
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0

        for agent in self.agents:
            total_action_space = []


            for _ in range(self.num_content_F):
                total_action_space.append(spaces.Discrete(2))

            for _ in range(self.world.num_users):
                total_action_space.append(spaces.Discrete(3))

            for _ in range(self.world.num_users):
                total_action_space.append(spaces.Discrete(self.dim_p))

            for _ in range(self.world.num_users):
                total_action_space.append(spaces.Discrete(self.dim_w))

            action_dims = [space.n for space in total_action_space]

            agent.action_dims = action_dims
            total_combinations = int(np.prod(action_dims))
            self.action_space.append(spaces.MultiDiscrete(action_dims))

            obs_dim = len(observation_callback(agent, self.world))
            share_obs_dim += obs_dim
            self.observation_space.append(spaces.Box(
                low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))

        self.share_observation_space = [spaces.Box(
            low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32)] * self.num_agents
    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def step(self, action_n):
        self.current_step += 1
        obs_n = []
        reward_n = []
        delay_n = []
        energy_n = []
        content_hit_n = []
        qoe_n = []

        done_n = []
        info_n = []
        self.agents = self.world.agents
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])

        self.world.step()

        for i, agent in enumerate(self.agents):
            obs_n.append(self._get_obs(agent))

            values = self._get_reward(agent)

            reward_n.append([values[0]])
            delay_n.append([values[1]])
            energy_n.append([values[2]])
            content_hit_n.append([values[3]])
            qoe_n.append([values[4]])

            done_n.append([self._get_done(agent)])

            info = {
                'individual_reward': values[0],
                'delay': values[1],
                'energy': values[2],
                'content_hit': values[3],
                'qoe': values[4],
            }
            if hasattr(agent, 'last_reward_info') and agent.last_reward_info is not None:
                info.update(agent.last_reward_info)
            info_n.append(info)

        return obs_n, reward_n, done_n, info_n, delay_n, energy_n, content_hit_n, qoe_n

    def reset(self):
        self.current_step = 0
        self.reset_callback(self.world)
        obs_n = []
        self.agents = self.world.agents

        for agent in self.agents:
            obs_n.append(self._get_obs(agent))

        return obs_n

    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    def _get_done(self, agent):
        if self.done_callback is None:
            if self.current_step >= self.world_length:
                return True
            else:
                return False
        return self.done_callback(agent, self.world)

    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    def _set_action(self, action, agent, action_space, time=None):
        decoded_action = np.array(action, dtype=int).flatten()

        expected_dim = self.world.num_content_F + 3 * self.world.num_users
        if len(decoded_action) != expected_dim:
            raise ValueError(
                f"Action dimension mismatch: got {len(decoded_action)}, expected {expected_dim}"
            )

        idx = 0

        agent.action.content_cache_action = decoded_action[idx: idx + self.world.num_content_F].astype(np.int32)
        idx += self.world.num_content_F

        agent.action.task_offload_action = decoded_action[idx: idx + self.world.num_users].astype(np.int32)
        idx += self.world.num_users

        offloads = agent.action.task_offload_action

        raw_comp = decoded_action[idx: idx + self.world.num_users].astype(np.float32) + 1.0
        idx += self.world.num_users

        comp = np.zeros(self.world.num_users, dtype=np.float32)
        edge_mask = (offloads == 1)

        if np.any(edge_mask):
            comp_raw_edge = raw_comp[edge_mask]
            comp[edge_mask] = comp_raw_edge / comp_raw_edge.sum()

        agent.action.computing_resource_action = comp

        raw_bw = decoded_action[idx: idx + self.world.num_users].astype(np.float32) + 1.0

        bw = np.zeros(self.world.num_users, dtype=np.float32)
        tx_mask = (offloads > 0)

        if np.any(tx_mask):
            bw_raw_tx = raw_bw[tx_mask]
            bw[tx_mask] = bw_raw_tx / bw_raw_tx.sum()

        agent.action.bandwidth_resource_action = bw
