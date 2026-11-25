import numpy as np
from offpolicy.envs.mpe.core import World, Agent, User
from offpolicy.envs.mpe.scenario import BaseScenario
import pandas as pd

pd.options.display.float_format = '{:.10f}'.format


class Scenario(BaseScenario):
    def _get_region_qoe_weights(self, agent, world):
        weights = np.asarray(
            getattr(world, "region_qoe_weights", [[world.q_t, world.q_e]]),
            dtype=np.float64,
        )
        if weights.ndim != 2 or weights.shape[1] != 2 or len(weights) == 0:
            return float(world.q_t), float(world.q_e)

        region_id = int(np.clip(agent.serial_number, 0, len(weights) - 1))
        q_t, q_e = np.clip(weights[region_id], 0.0, None)
        total = float(q_t + q_e)
        if total <= 0.0:
            return float(world.q_t), float(world.q_e)
        return float(q_t / total), float(q_e / total)

    def _nearest_bs_id(self, user_location, bs_location_set):
        distances = []
        for bs_loc in bs_location_set:
            dx = user_location[0] - bs_loc[0]
            dy = user_location[1] - bs_loc[1]
            distances.append(np.sqrt(dx * dx + dy * dy))
        return int(np.argmin(distances))

    def _build_regional_zipf_popularity(
            self,
            num_contents,
            content_rank,
            associated_bs,
            num_agents,
            delta=2.0,
            boost_strength=20.0,
            block_size=30,
            stride=20
    ):
        rank = np.arange(1, num_contents + 1, dtype=np.float64)
        base_pop = 1.0 / np.power(rank, delta)
        boost = np.ones(num_contents, dtype=np.float64)

        start = (associated_bs * stride) % num_contents
        indices = np.arange(start, start + block_size) % num_contents
        boost[indices] *= boost_strength

        pop = base_pop * boost
        pop = pop / pop.sum()
        return pop.tolist()

    def make_world(self, args):
        world = World()
        world.world = world

        world.cache_update_interval = int(getattr(args, "cache_update_interval", 5))
        world.preference_update_interval = int(getattr(args, "preference_update_interval", 100000))
        world.cache_hit_reward = float(getattr(args, "cache_hit_reward", 200.0))
        world.cache_miss_penalty = float(getattr(args, "cache_miss_penalty", 50.0))
        world.qoe_penalty_weight = float(getattr(args, "qoe_penalty_weight", 220.0))
        world.edge_cache_delay_gain = float(getattr(args, "edge_cache_delay_gain", 0.30))
        world.edge_cache_energy_gain = float(getattr(args, "edge_cache_energy_gain", 0.25))
        world.preference_noise = float(getattr(args, "preference_noise", 0.01))
        world.preserve_cache_on_reset = bool(getattr(args, "preserve_cache_on_reset", True))
        world.cache_policy_blend = float(getattr(args, "cache_policy_blend", 0.95))
        world.cache_request_window = int(getattr(args, "cache_request_window", 1000))
        world.cache_curriculum_episodes = int(getattr(args, "cache_curriculum_episodes", 200))
        world.episode_steps_for_curriculum = int(getattr(args, "max_steps", 50))
        world.time_slots_per_day = int(getattr(args, "time_slots_per_day", 24))
        world.demand_event_probability = float(getattr(args, "demand_event_probability", 0.04))
        world.demand_event_duration = int(getattr(args, "demand_event_duration", 6))
        world.demand_event_strength = float(getattr(args, "demand_event_strength", 0.35))
        world.spatial_demand_correlation = float(getattr(args, "spatial_demand_correlation", 0.25))
        world.regional_demand_strength = float(getattr(args, "regional_demand_strength", 1.0))
        world.resource_heterogeneity_strength = float(getattr(args, "resource_heterogeneity_strength", 1.0))
        world.service_temporal_weight = float(getattr(args, "service_temporal_weight", 0.40))
        world.qoe_reward_weight = float(getattr(args, "qoe_reward_weight", 260.0))
        world.delay_penalty_weight = float(getattr(args, "delay_penalty_weight", 120.0))
        world.energy_penalty_weight = float(getattr(args, "energy_penalty_weight", 20.0))
        world.overtime_penalty_weight = float(getattr(args, "overtime_penalty_weight", 180.0))
        world.qoe_delay_slope = float(getattr(args, "qoe_delay_slope", 1.4))
        world.edge_collab_bonus = float(getattr(args, "edge_collab_bonus", 40.0))

        world.num_agents = 5
        world.num_users = 30
        world.num_service_types = 4
        world.alpha = 2
        world.N0 = 1e-13

        world.q_t = 0.6
        world.q_e = 0.4

        region_q_t = np.asarray(
            getattr(args, "region_q_t", (0.60, 0.75, 0.85, 0.80, 0.55)),
            dtype=np.float64,
        ).reshape(-1)
        region_q_e = np.asarray(
            getattr(args, "region_q_e", (0.40, 0.25, 0.15, 0.20, 0.45)),
            dtype=np.float64,
        ).reshape(-1)
        if len(region_q_t) != world.num_agents or len(region_q_e) != world.num_agents:
            raise ValueError(
                f"region_q_t and region_q_e must each contain {world.num_agents} values"
            )
        if np.any(region_q_t < 0.0) or np.any(region_q_e < 0.0):
            raise ValueError("region_q_t and region_q_e must be non-negative")
        region_weight_sums = region_q_t + region_q_e
        if np.any(region_weight_sums <= 0.0):
            raise ValueError("Each region must have at least one positive QoE weight")
        world.region_qoe_weights = np.stack(
            [region_q_t / region_weight_sums, region_q_e / region_weight_sums],
            axis=1,
        )

        world.service_params = {
            0: dict(name="video_transcoding", size=8e7, cycles=5e8, delay_fair=2.0, delay_max=4.0,
                    energy_ref=12.0, qoe_delay_slope=5.0, qoe_energy_sensitivity=1.0),
            1: dict(name="ar_vr_interaction", size=2e7, cycles=2e9, delay_fair=0.20, delay_max=0.5,
                    energy_ref=10.0, qoe_delay_slope=12.0, qoe_energy_sensitivity=0.8),
            2: dict(name="mobile_vision_analysis", size=5e7, cycles=1e9, delay_fair=0.50, delay_max=1.0,
                    energy_ref=14.0, qoe_delay_slope=9.0, qoe_energy_sensitivity=1.1),
            3: dict(name="content_recommendation", size=3e7, cycles=3e8, delay_fair=4.0, delay_max=8.0,
                    energy_ref=8.0, qoe_delay_slope=4.0, qoe_energy_sensitivity=1.2),
        }

        world.cloud_backhaul_rate = 10.0
        world.bs_link_rate = np.ones((world.num_agents, world.num_agents)) * 50.0

        BS_max_content_cached = int(getattr(args, "bs_cache_capacity", 35))

        BS_location_set = [
            (100, 500), (350, 300), (600, 700), (850, 400), (550, 600)
        ]
        world.bs_positions = BS_location_set
        bs_xy = np.asarray(BS_location_set, dtype=np.float32)
        bs_dist = np.linalg.norm(bs_xy[:, None, :] - bs_xy[None, :, :], axis=-1)
        bs_adj = np.zeros((world.num_agents, world.num_agents), dtype=np.float32)
        for i in range(world.num_agents):
            neighbor_idx = np.argsort(bs_dist[i])[1: min(world.num_agents, 4)]
            bs_adj[i, neighbor_idx] = 1.0
        bs_adj = np.maximum(bs_adj, bs_adj.T)
        np.fill_diagonal(bs_adj, 1.0)
        world.bs_adjacency = bs_adj

        world.region_service_profiles = np.asarray([
            [0.38, 0.18, 0.24, 0.20],
            [0.25, 0.18, 0.38, 0.19],
            [0.30, 0.30, 0.22, 0.18],
            [0.24, 0.16, 0.45, 0.15],
            [0.42, 0.22, 0.14, 0.22],
        ], dtype=np.float64)
        world.temporal_service_profiles = np.asarray([
            [0.42, 0.10, 0.16, 0.32],
            [0.30, 0.18, 0.34, 0.18],
            [0.24, 0.18, 0.40, 0.18],
            [0.40, 0.30, 0.12, 0.18],
        ], dtype=np.float64)
        world.num_content_F = int(getattr(args, "num_content_F", 100))
        world.region_content_shifts = np.arange(world.num_agents) * max(1, int(world.num_content_F * 0.11))
        world.active_events = [None for _ in range(world.num_agents)]

        ED_location_set = [
            (200, 400), (250, 550), (700, 600), (800, 700), (300, 450),
            (120, 520), (180, 280), (320, 680), (680, 380), (920, 620),
            (760, 820), (420, 520), (540, 580), (610, 720), (390, 320),
            (150, 460), (280, 360), (720, 560), (850, 760), (360, 480),
            (450, 350), (500, 450), (650, 500), (780, 300), (220, 620),
            (480, 680), (580, 400), (820, 550), (320, 350), (900, 450)
        ]

        all_x = [p[0] for p in BS_location_set + ED_location_set]
        all_y = [p[1] for p in BS_location_set + ED_location_set]
        world.max_distance = np.sqrt(
            (max(all_x) - min(all_x)) ** 2 + (max(all_y) - min(all_y)) ** 2
        )

        world.agents = [Agent() for _ in range(world.num_agents)]

        resource_strength = float(np.clip(getattr(world, "resource_heterogeneity_strength", 1.0), 0.0, 4.0))

        def amplify_resource_scales(scales):
            arr = np.asarray(scales, dtype=np.float64)
            center = float(arr.mean())
            arr = center + (arr - center) * resource_strength
            return np.clip(arr, 0.2, None).tolist()

        bandwidth_scales = amplify_resource_scales([1.50, 0.60, 1.30, 0.80, 1.10])
        compute_scales = amplify_resource_scales([0.60, 1.80, 0.90, 1.50, 1.10])
        cache_scales = amplify_resource_scales([1.00, 0.80, 1.00, 0.90, 1.20])

        for i, agent in enumerate(world.agents):
            agent.name = f'BS {i}'
            agent.serial_number = i
            agent.location = BS_location_set[i]
            agent.state.location_state = BS_location_set[i]

            base_bandwidth =50.0
            base_computing = float(getattr(args, "bs_compute_capacity", 50e9))

            agent.max_content_cached = BS_max_content_cached

            agent.total_bandwidth = base_bandwidth * bandwidth_scales[i]
            agent.state.available_bandwidth = agent.total_bandwidth

            agent.total_computing_resource = base_computing * compute_scales[i]
            agent.state.available_computing = agent.total_computing_resource

            agent.max_content_cached = max(1, int(agent.max_content_cached * cache_scales[i]))

            agent.operation_power = 2.0
            agent.power_backbone_max_min = 0.5

        world.max_bandwidth_resource = max(agent.total_bandwidth for agent in world.agents)
        world.max_computing_resource = max(agent.total_computing_resource for agent in world.agents)
        world.max_cache_capacity = max(agent.max_content_cached for agent in world.agents)

        world.users = [User() for _ in range(world.num_users)]

        world.num_content_F = int(getattr(args, "num_content_F", 100))
        delta1 = 1.3
        delta2 = 0.6

        service_rank = list(range(1, world.num_content_F + 1))
        rank = np.arange(1, world.num_content_F + 1, dtype=np.float64)

        base_pop_global = 1.0 / np.power(rank, delta1)
        base_pop_global /= base_pop_global.sum()

        base_pop_flat = 1.0 / np.power(rank, delta2)
        base_pop_flat /= base_pop_flat.sum()

        for i, user in enumerate(world.users):
            user.name = f'ED {i}'
            user.serial_number = i
            user.location = ED_location_set[i]
            user.content_rank = service_rank

            associated_bs = self._nearest_bs_id(user.location, BS_location_set)
            user.associated_bs = associated_bs
            user.service_profile = world.region_service_profiles[associated_bs].copy()

            regional_strength = float(np.clip(getattr(world, "regional_demand_strength", 1.0), 0.1, 5.0))
            regional_block_size = max(
                10,
                int(world.num_content_F * float(np.clip(0.40 / np.sqrt(regional_strength), 0.18, 0.45)))
            )
            regional_stride = max(
                1,
                int(world.num_content_F * float(np.clip(0.15 * regional_strength, 0.15, 0.40)))
            )
            regional_boost_strength = 200.0 * regional_strength

            user.regional_profile = self._build_regional_zipf_popularity(
                num_contents=world.num_content_F,
                content_rank=service_rank,
                associated_bs=associated_bs,
                num_agents=world.num_agents,
                delta=1.2,
                boost_strength=regional_boost_strength,
                block_size=regional_block_size,
                stride=regional_stride
            )

            user.global_profile = base_pop_global.tolist()

            user.pop_profile_vector1 = user.regional_profile.copy()
            user.pop_profile_vector2 = user.global_profile.copy()

            user.pop_profile_vector = user.global_profile.copy()

            user.preference_level = 0

            user.total_local_computing = 4.5e8
            user.local_computing = user.total_local_computing
            user.transmit_power = 1.0
            user.capacitance_coefficient = 8e-26
            user.task_size = 5e7
            user.task_cycles = 1e9
            user.task_delay_max = 5.0

        world.regional_phases = [False] * world.num_agents

        self.reset_world(world)
        return world

    def reset_world(self, world):
        preserve_cache = bool(getattr(world, 'preserve_cache_on_reset', True))
        for agent in world.agents:
            valid_cache = (
                preserve_cache
                and agent.state.content_state is not None
                and len(agent.state.content_state) == world.num_content_F
                and int(np.sum(agent.state.content_state)) > 0
            )
            if valid_cache:
                agent.state.content_state = np.asarray(agent.state.content_state, dtype=int)
            else:
                cache_capacity = min(int(agent.max_content_cached), world.num_content_F)
                if hasattr(world, 'build_cache_state'):
                    agent.state.content_state = world.build_cache_state(
                        agent, np.zeros(world.num_content_F, dtype=np.int32)
                    )
                else:
                    agent.state.content_state = np.zeros(world.num_content_F, dtype=int)
                    indices = np.random.choice(world.num_content_F, cache_capacity, replace=False)
                    agent.state.content_state[indices] = 1
            agent.state.previous_cache_state = agent.state.content_state.copy()
            agent.real_state.content_state = agent.state.content_state.copy()

            dis = np.zeros(world.num_users)
            for i, user in enumerate(world.users):
                dis[i] = self._get_distance(agent, user)
            agent.state.agent_user_distance_state = dis

            agent.state.available_bandwidth = agent.total_bandwidth
            agent.state.available_computing = agent.total_computing_resource

        if getattr(world, 'world_step', 0) == 0:
            for i in range(world.num_agents):
                world.regional_phases[i] = True

        for user in world.users:
            if hasattr(user, 'global_profile') and user.global_profile:
                user.pop_profile_vector = user.global_profile.copy()
            else:
                user.pop_profile_vector = user.pop_profile_vector1.copy()
            if hasattr(world, '_update_user_multimedia_demand'):
                world._update_user_multimedia_demand(user)

    def _get_distance(self, agent, user):
        return np.sqrt((agent.state.location_state[0] - user.location[0]) ** 2 +
                       (agent.state.location_state[1] - user.location[1]) ** 2)

    def observation(self, agent, world):
        obs_cache = agent.state.content_state.astype(np.float32)

        obs_req = []
        obs_association = []
        for user in world.users:
            req = getattr(user.state, 'user_content_req', 0)
            obs_req.append((req - 1) / max(world.num_content_F - 1, 1.0) if req else 0.0)
            obs_association.append(
                1.0 if getattr(user, 'associated_bs', agent.serial_number) == agent.serial_number else 0.0)

        obs_req = np.array(obs_req, dtype=np.float32)
        obs_association = np.array(obs_association, dtype=np.float32)
        obs_distance = np.array(agent.state.agent_user_distance_state, dtype=np.float32) / max(world.max_distance, 1e-6)
        obs_bandwidth = np.array([agent.state.available_bandwidth / max(agent.total_bandwidth, 1e-6)], dtype=np.float32)
        obs_computing = np.array([agent.state.available_computing / max(agent.total_computing_resource, 1e-6)],
                                 dtype=np.float32)
        obs_resource_profile = np.array([
            agent.total_bandwidth / max(getattr(world, 'max_bandwidth_resource', agent.total_bandwidth), 1e-6),
            agent.total_computing_resource / max(getattr(world, 'max_computing_resource', agent.total_computing_resource), 1e-6),
            agent.max_content_cached / max(world.num_content_F, 1),
        ], dtype=np.float32)

        phases = getattr(world, 'regional_phases', None)
        agent_phase = float(phases[agent.serial_number]) if phases is not None else 0.0
        obs_phase = np.array([agent_phase], dtype=np.float32)

        assoc_count = sum(
            1 for u in world.users
            if getattr(u, 'associated_bs', agent.serial_number) == agent.serial_number
        )
        obs_assoc_count = np.array([assoc_count / max(world.num_users, 1)], dtype=np.float32)

        svc_dist = np.zeros(getattr(world, 'num_service_types', 4), dtype=np.float32)
        svc_served = 0
        for user in world.users:
            if getattr(user, 'associated_bs', agent.serial_number) == agent.serial_number:
                svc_dist[getattr(user, 'service_type', 0)] += 1.0
                svc_served += 1
        if svc_served > 0:
            svc_dist /= svc_served
        obs_svc_dist = svc_dist
        if hasattr(world, 'get_agent_service_mix'):
            obs_svc_mix = np.asarray(world.get_agent_service_mix(agent), dtype=np.float32)
        else:
            obs_svc_mix = obs_svc_dist

        if hasattr(world, '_temporal_phase_index'):
            _, slot = world._temporal_phase_index()
        else:
            slot = getattr(world, 'world_step', 0)
        slots_per_day = max(1, int(getattr(world, 'time_slots_per_day', 24)))
        time_angle = 2.0 * np.pi * (slot / float(slots_per_day))
        obs_time = np.array([np.sin(time_angle), np.cos(time_angle)], dtype=np.float32)

        active_events = getattr(world, 'active_events', None)
        event_flag = 0.0
        if active_events is not None and agent.serial_number < len(active_events):
            event_flag = 1.0 if active_events[agent.serial_number] is not None else 0.0
        obs_event = np.array([event_flag], dtype=np.float32)

        pop = world.get_agent_popularity(agent)
        obs_pop = np.asarray(pop, dtype=np.float32)

        return np.concatenate([obs_cache, obs_req, obs_distance, obs_association,
                               obs_bandwidth, obs_computing, obs_resource_profile, obs_phase, obs_assoc_count,
                               obs_svc_dist, obs_svc_mix, obs_time, obs_event, obs_pop])

    def _legacy_reward(self, agent, world):
        served_users = [
            u for u in world.users
            if getattr(u, 'associated_bs', agent.serial_number) == agent.serial_number
        ]
        if not served_users:
            served_users = world.users

        total_delay = 0.0
        total_energy = 0.0
        total_hit = 0
        total_edge_hit = 0
        total_cost = 0.0
        total_overtime = 0

        for u in served_users:
            _, delay, energy, hit, edge_hit = self._compute_task_cost(agent, u, world)
            total_delay += delay
            total_energy += energy
            total_hit += hit
            total_edge_hit += edge_hit

            svc = getattr(u, 'service_type', 0)
            sp = world.service_params.get(svc, world.service_params[0])
            delay_max = sp['delay_max']

            delay_norm = min(delay / max(delay_max, 1e-6), 5.0)
            energy_norm = energy / 10.0
            overtime = 1.0 if delay > delay_max else 0.0
            total_overtime += overtime

            q_t, q_e = self._get_region_qoe_weights(agent, world)
            cost_u = q_t * delay_norm + q_e * energy_norm + overtime
            total_cost += cost_u

        n = max(len(served_users), 1)
        avg_cost = total_cost / n
        avg_delay = total_delay / n
        avg_energy = total_energy / n
        hit_rate = total_hit / n
        edge_hit_rate = total_edge_hit / n
        qoe_loss = total_overtime / n

        cost_penalty = -220.0 * avg_cost

        cache_hit_reward = float(getattr(world, 'cache_hit_reward', 300.0))
        cache_miss_penalty = float(getattr(world, 'cache_miss_penalty', 80.0))
        hit_bonus = cache_hit_reward * (0.25 * hit_rate + 0.45 * edge_hit_rate)
        miss_penalty = cache_miss_penalty * (1.0 - hit_rate)
        qoe_penalty = float(getattr(world, 'qoe_penalty_weight', 220.0)) * qoe_loss

        reward_value = cost_penalty + hit_bonus - miss_penalty - qoe_penalty
        reward_value = np.clip(reward_value, -500.0, 300.0)

        agent.last_reward_info = {
            'reward': float(reward_value),
            'avg_cost': float(avg_cost),
            'hit_rate': float(hit_rate),
            'edge_hit_rate': float(edge_hit_rate),
            'avg_delay': float(avg_delay),
            'avg_energy': float(avg_energy),
            'overtime_ratio': float(qoe_loss),
        }

        return float(reward_value), float(avg_delay), float(avg_energy), float(hit_rate), float(qoe_loss)


    def reward(self, agent, world):
        served_users = [
            u for u in world.users
            if getattr(u, 'associated_bs', agent.serial_number) == agent.serial_number
        ]
        if not served_users:
            served_users = world.users

        total_delay = 0.0
        total_energy = 0.0
        total_hit = 0.0
        total_edge_hit = 0.0
        total_delay_norm = 0.0
        total_energy_norm = 0.0
        total_overtime = 0.0
        total_qoe = 0.0
        total_delay_utility = 0.0
        total_energy_utility = 0.0

        q_t, q_e = self._get_region_qoe_weights(agent, world)

        for u in served_users:
            _, delay, energy, hit, edge_hit = self._compute_task_cost(agent, u, world)
            total_delay += delay
            total_energy += energy
            total_hit += hit
            total_edge_hit += edge_hit

            svc = getattr(u, 'service_type', 0)
            sp = world.service_params.get(svc, world.service_params[0])
            delay_max = float(sp['delay_max'])
            energy_ref = float(sp.get('energy_ref', 10.0))

            delay_ratio = max(0.0, delay / max(delay_max, 1e-6))
            energy_ratio = max(0.0, energy / max(energy_ref, 1e-6))
            delay_norm = min(delay_ratio, 5.0)
            energy_norm = min(energy_ratio, 5.0)
            overtime = 1.0 if delay > delay_max else 0.0

            delay_slope = float(sp.get('qoe_delay_slope', getattr(world, 'qoe_delay_slope', 1.4)))
            energy_sensitivity = float(sp.get('qoe_energy_sensitivity', 1.0))
            sigmoid_arg = float(np.clip(delay_slope * (delay_ratio - 1.0), -60.0, 60.0))
            energy_exponent = float(np.clip(energy_sensitivity * energy_ratio, 0.0, 60.0))
            delay_utility = float(1.0 / (1.0 + np.exp(sigmoid_arg)))
            energy_utility = float(np.exp(-energy_exponent))
            qoe_u = q_t * delay_utility + q_e * energy_utility

            total_delay_norm += delay_norm
            total_energy_norm += energy_norm
            total_overtime += overtime
            total_delay_utility += delay_utility
            total_energy_utility += energy_utility
            total_qoe += float(np.clip(qoe_u, 0.0, 1.0))

        n = max(len(served_users), 1)
        avg_delay = total_delay / n
        avg_energy = total_energy / n
        hit_rate = total_hit / n
        edge_hit_rate = total_edge_hit / n
        neighbor_hit_rate = max(0.0, edge_hit_rate - hit_rate)
        avg_delay_norm = total_delay_norm / n
        avg_energy_norm = total_energy_norm / n
        overtime_ratio = total_overtime / n
        avg_qoe = total_qoe / n
        avg_delay_utility = total_delay_utility / n
        avg_energy_utility = total_energy_utility / n

        qoe_cost = 1.0 - avg_qoe
        reward_value = -qoe_cost

        agent.last_reward_info = {
            'reward': float(reward_value),
            'qoe_score': float(avg_qoe),
            'qoe_cost': float(qoe_cost),
            'avg_cost': float(qoe_cost),
            'delay_utility': float(avg_delay_utility),
            'energy_utility': float(avg_energy_utility),
            'region_q_t': float(q_t),
            'region_q_e': float(q_e),
            'hit_rate': float(hit_rate),
            'edge_hit_rate': float(edge_hit_rate),
            'neighbor_hit_rate': float(neighbor_hit_rate),
            'avg_delay': float(avg_delay),
            'avg_energy': float(avg_energy),
            'delay_norm': float(avg_delay_norm),
            'energy_norm': float(avg_energy_norm),
            'overtime_ratio': float(overtime_ratio),
        }

        return float(reward_value), float(avg_delay), float(avg_energy), float(hit_rate), float(avg_qoe)

    def _compute_task_cost(self, agent, user, world):
        offload_decision = self._get_offload_decision(agent, user)
        bandwidth_ratio = self._get_bandwidth_ratio(agent, user)
        computing_ratio = self._get_computing_ratio(agent, user)

        service_id = user.state.user_content_req - 1
        service_cached = agent.state.content_state[service_id]

        svc_type = getattr(user, 'service_type', 0)
        svc_params = world.service_params.get(svc_type, world.service_params[0])
        task_size = svc_params['size']
        task_cycles = svc_params['cycles']
        task_delay_max = svc_params['delay_max']

        upload_rate = self._compute_upload_rate(agent, user, bandwidth_ratio, world)
        upload_delay = task_size / max(upload_rate, 1e-6)
        upload_energy = user.transmit_power * upload_delay

        cache_hit = int(service_cached > 0)
        execution_hit = 0

        if offload_decision == 0:
            compute_delay = task_cycles / max(user.total_local_computing, 1e-6)
            compute_energy = (
                    user.capacitance_coefficient
                    * task_cycles
                    * (user.total_local_computing ** 2)
            )
            total_delay = compute_delay
            total_energy = compute_energy
        elif offload_decision == 1:
            total_delay, total_energy, execution_hit = self._edge_computing(
                agent, user, world, upload_delay, upload_energy,
                service_cached, computing_ratio, task_size, task_cycles
            )
        else:
            total_delay, total_energy, execution_hit = self._cloud_computing(
                agent, user, world, upload_delay, upload_energy, task_size
            )

        if execution_hit == 1:
            total_steps = max(1, int(getattr(world, 'episode_counter', 0)))
            steps_per_episode = max(1, int(getattr(world, 'episode_steps_for_curriculum', 50)))
            curriculum_episodes = max(1.0, float(getattr(world, 'cache_curriculum_episodes', 200)))
            cache_progress = min(1.0, total_steps / (steps_per_episode * curriculum_episodes))
            delay_gain = float(np.clip(getattr(world, 'edge_cache_delay_gain', 0.30), 0.0, 0.60))
            energy_gain = float(np.clip(getattr(world, 'edge_cache_energy_gain', 0.25), 0.0, 0.60))
            total_delay *= (1.0 - delay_gain * cache_progress)
            total_energy *= (1.0 - energy_gain * cache_progress)

        q_t, q_e = self._get_region_qoe_weights(agent, world)
        energy_ref = float(svc_params.get('energy_ref', 10.0))
        delay_norm = np.clip(total_delay / max(task_delay_max, 1e-6), 0.0, 20.0)
        energy_norm = np.clip(total_energy / max(energy_ref, 1e-6), 0.0, 20.0)
        cost = q_t * delay_norm + q_e * energy_norm

        return float(cost), float(total_delay), float(total_energy), int(cache_hit), int(execution_hit)

    def _edge_computing(self, agent, user, world, upload_delay, upload_energy,
                        service_cached, computing_ratio, task_size=None, task_cycles=None):
        _size = task_size if task_size is not None else user.task_size
        _cycles = task_cycles if task_cycles is not None else user.task_cycles
        service_id = user.state.user_content_req - 1

        if service_cached and computing_ratio > 0:
            allocated_resource = max(computing_ratio * agent.total_computing_resource, 1e-6)
            compute_delay = _cycles / allocated_resource
            compute_energy = agent.operation_power * compute_delay * 0.3
            total_energy = upload_energy + compute_energy
            return upload_delay + compute_delay, total_energy, 1

        coordination_penalty = 0.2

        neighbor_caches = np.array([other.state.content_state[service_id] for other in world.agents])
        neighbor_caches[agent.serial_number] = 0

        if np.any(neighbor_caches):
            neighbor_ids = np.where(neighbor_caches == 1)[0]
            best_delay = float("inf")
            best_result = None

            for other_id in neighbor_ids:
                other_agent = world.agents[other_id]

                bs_link_rate = max(world.bs_link_rate[agent.serial_number, other_id], 1e-6)
                bs_link_delay = _size / (bs_link_rate * 1e6)

                allocated_resource = max(computing_ratio * other_agent.total_computing_resource, 1e-6)
                compute_delay = _cycles / allocated_resource

                total_delay = upload_delay + bs_link_delay + compute_delay + coordination_penalty

                if total_delay < best_delay:
                    best_delay = total_delay
                    best_result = (total_delay, upload_energy, 1)

            return best_result

        c_delay, c_energy, c_hit = self._cloud_computing(
            agent, user, world, upload_delay, upload_energy, _size)
        return c_delay, c_energy, c_hit

    def _cloud_computing(self, agent, user, world, upload_delay, upload_energy, task_size=None):
        _size = task_size if task_size is not None else user.task_size
        backhaul_time = _size / (max(world.cloud_backhaul_rate * 0.5, 1e-6) * 1e6) + 3.0
        total_delay = upload_delay + backhaul_time
        backhaul_energy = agent.power_backbone_max_min * backhaul_time
        total_energy = upload_energy * 2.0 + backhaul_energy
        return total_delay, total_energy, 0

    def _compute_upload_rate(self, agent, user, bandwidth_ratio, world):
        distance = max(self._get_distance(agent, user), 1.0)
        channel_gain = distance ** (-world.alpha)

        bandwidth_ratio = np.clip(bandwidth_ratio, 0.01, 1.0)
        allocated_bandwidth = max(bandwidth_ratio * agent.total_bandwidth * 1e6, 1e3)

        snr = (user.transmit_power * channel_gain) / max(world.N0, 1e-15)
        snr = np.clip(snr, 1e-6, 1e4)

        rate = allocated_bandwidth * np.log2(1.0 + snr)
        return max(rate, 1e-6)

    def _get_offload_decision(self, agent, user):
        if hasattr(agent.action, 'task_offload_action') and agent.action.task_offload_action is not None:
            return agent.action.task_offload_action[user.serial_number]
        return 1

    def _get_bandwidth_ratio(self, agent, user):
        if hasattr(agent.action, 'bandwidth_resource_action') and agent.action.bandwidth_resource_action is not None:
            return float(agent.action.bandwidth_resource_action[user.serial_number])
        return 1.0 / len(agent.state.agent_user_distance_state)

    def _get_computing_ratio(self, agent, user):
        if hasattr(agent.action, 'computing_resource_action') and agent.action.computing_resource_action is not None:
            return float(agent.action.computing_resource_action[user.serial_number])
        return 1.0 / len(agent.state.agent_user_distance_state)
