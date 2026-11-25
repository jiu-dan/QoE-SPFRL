import numpy as np

from myUnits import random_pick


class AgentState(object):
    def __init__(self):
        self.user_location_state = None
        self.agent_user_distance_state = None
        self.user_content_req = None
        self.content_state = None
        self.previous_cache_state = None
        self.available_bandwidth = None
        self.available_computing = None
        self.location_state = None
        self.terminal_info_state = None


class UserState(object):
    def __init__(self):
        self.user_content_req = None
        self.available_local_computing = None


class RealState(object):
    def __init__(self):
        self.content_state = None
        self.location_user = None


class Action(object):
    def __init__(self):
        self.content_cache_action = None
        self.task_offload_action = None
        self.computing_resource_action = None
        self.bandwidth_resource_action = None


class User(object):
    def __init__(self):
        self.name = ''
        self.serial_number = None
        self.location = None
        self.content_rank = None
        self.delta = None
        self.state = UserState()

        self.task_size = None
        self.task_cycles = None
        self.task_delay_max = None

        self.local_computing = None
        self.total_local_computing = None
        self.used_local_computing = 0.0
        self.transmit_power = None
        self.capacitance_coefficient = None

        self.pop_profile_vector = []
        self.pop_profile_vector1 = []
        self.pop_profile_vector2 = []
        self.pop_profile_vector3 = []
        self.pop_profile_vector4 = []
        self.global_profile = []
        self.regional_profile = []

        self.associated_bs = 0
        self.preference_level = 0
        self.service_type = 0
        self.service_profile = None
        self.current_service_profile = None


class Agent(object):
    def __init__(self):
        self.name = ''
        self.serial_number = None
        self.location = None

        self.max_content_cached = None
        self.delta = None
        self.pop_profile_vector = None

        self.total_bandwidth = None
        self.total_computing_resource = None
        self.operation_power = None
        self.power_backbone_max_min = None

        self.used_bandwidth = 0.0
        self.used_computing_resource = 0.0

        self.state = AgentState()
        self.real_state = RealState()
        self.action = Action()
        self.action_callback = None


class World(object):
    def __init__(self):
        self.agents = []
        self.num_agents = None
        self.users = []
        self.num_users = None
        self.num_content_F = None
        self.num_service_types = 4

        self.cache_update_interval = 50
        self.preference_update_interval = 100000
        self.cache_policy_blend = 0.85
        self.cache_request_window = 1000
        self.cache_curriculum_episodes = 200
        self.episode_steps_for_curriculum = 50
        self.request_history = None
        self.service_history = None

        self.N0 = 1e-13
        self.theta_pen = None
        self.dim_p = 10
        self.dim_w = 10
        self.world_length = 100
        self.world_step = 0
        self.episode_counter = 0

        self.cloud_total_computing = None
        self.cloud_used_computing = 0.0
        self.cloud_backhaul_rate = None
        self.backhaul_total_bandwidth = None
        self.backhaul_used_bandwidth = 0.0
        self.ap_link_total_bandwidth = None
        self.ap_link_used_bandwidth = None
        self.bs_link_rate = None
        self.alpha = 2

        self.time_slots_per_day = 24
        self.demand_event_probability = 0.04
        self.demand_event_duration = 6
        self.demand_event_strength = 0.35
        self.spatial_demand_correlation = 0.25
        self.service_temporal_weight = 0.40
        self.region_service_profiles = None
        self.temporal_service_profiles = None
        self.bs_adjacency = None
        self.active_events = None

    def step(self):
        self.world_step += 1
        self.episode_counter += 1

        for agent in self.agents:
            agent.used_bandwidth = 0.0
            agent.used_computing_resource = 0.0
        self.cloud_used_computing = 0.0
        self.backhaul_used_bandwidth = 0.0

        for user in self.users:
            user.used_local_computing = 0.0

        self._advance_spatiotemporal_events()
        preference_interval = max(1, int(getattr(self, 'preference_update_interval', 100000)))
        if self.world_step % preference_interval == 0:
            self.update_user_pop_profile()

        for user in self.users:
            self._update_user_multimedia_demand(user)

        self.update_request_history()

        agent_locs = np.array([agent.location for agent in self.agents], dtype=np.float32)
        user_locs = np.array([user.location for user in self.users], dtype=np.float32)
        dist_matrix = np.sqrt(((agent_locs[:, None, :] - user_locs[None, :, :]) ** 2).sum(axis=-1))

        for idx, agent in enumerate(self.agents):
            self.update_agent_state(agent, self, dist_matrix[idx])

    def update_agent_state(self, agent, world, distances_to_users):
        agent.state.previous_cache_state = (
            None if agent.state.content_state is None else np.copy(agent.state.content_state)
        )

        dis_array = distances_to_users.astype(np.float32)
        req_array = []

        for user in world.users:
            local_comp = user.local_computing if user.local_computing is not None else 0.0
            user.state.available_local_computing = max(0.0, local_comp - user.used_local_computing)
            req_array.append(user.state.user_content_req)

        agent.state.agent_user_distance_state = dis_array
        agent.state.user_location_state = [user.location for user in world.users]
        agent.state.user_content_req = req_array

        bw_ratios = np.asarray(
            agent.action.bandwidth_resource_action
            if agent.action.bandwidth_resource_action is not None
            else np.zeros(world.num_users),
            dtype=np.float32,
        )
        comp_ratios = np.asarray(
            agent.action.computing_resource_action
            if agent.action.computing_resource_action is not None
            else np.zeros(world.num_users),
            dtype=np.float32,
        )
        offloads = np.asarray(
            agent.action.task_offload_action
            if agent.action.task_offload_action is not None
            else np.ones(world.num_users),
            dtype=np.int32,
        )

        edge_mask = offloads == 1
        cloud_mask = offloads == 2
        tx_mask = offloads > 0

        used_bw_ratio = min(float(np.sum(bw_ratios[tx_mask])), 1.0)
        used_comp_ratio = min(float(np.sum(comp_ratios[edge_mask])), 1.0)
        world.backhaul_used_bandwidth += float(np.sum(bw_ratios[cloud_mask])) * agent.total_bandwidth

        agent.used_bandwidth = used_bw_ratio * agent.total_bandwidth
        agent.used_computing_resource = used_comp_ratio * agent.total_computing_resource
        agent.state.available_bandwidth = max(0.0, agent.total_bandwidth - agent.used_bandwidth)
        agent.state.available_computing = max(0.0, agent.total_computing_resource - agent.used_computing_resource)

        cache_interval = max(1, int(getattr(self, 'cache_update_interval', 50)))
        if (self.world_step - 1) % cache_interval == 0 and agent.action.content_cache_action is not None:
            new_cache = self.build_cache_state(agent, agent.action.content_cache_action)
            agent.state.content_state = new_cache
            agent.real_state.content_state = np.copy(new_cache)

    def _normalize_prob(self, values):
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return arr
        arr = np.where(np.isfinite(arr), arr, 0.0)
        arr = np.clip(arr, 0.0, None)
        total = arr.sum()
        if total <= 0.0:
            return np.ones(arr.size, dtype=np.float64) / arr.size
        return arr / total

    def _temporal_phase_index(self):
        slots = max(1, int(getattr(self, 'time_slots_per_day', 24)))
        slot = int(self.world_step % slots)
        phase = min(3, int(4 * slot / slots))
        return phase, slot

    def _advance_spatiotemporal_events(self):
        if self.active_events is None or len(self.active_events) != self.num_agents:
            self.active_events = [None for _ in range(self.num_agents)]

        event_prob = float(np.clip(getattr(self, 'demand_event_probability', 0.04), 0.0, 1.0))
        duration = max(1, int(getattr(self, 'demand_event_duration', 6)))

        for bs in range(self.num_agents):
            event = self.active_events[bs]
            if event is not None:
                event['remaining'] -= 1
                if event['remaining'] <= 0:
                    self.active_events[bs] = None

            if self.active_events[bs] is None and np.random.random() < event_prob:
                shifts = getattr(self, 'region_content_shifts', np.zeros(self.num_agents, dtype=np.int32))
                shift = int(shifts[bs]) if bs < len(shifts) else 0
                center = int((shift + 7 * self.world_step + 13 * bs) % max(self.num_content_F, 1))
                service_profile = self._region_service_profile(bs)
                service_type = int(np.random.choice(self.num_service_types, p=service_profile))
                self.active_events[bs] = {
                    'center': center,
                    'service_type': service_type,
                    'remaining': duration,
                }

    def _region_service_profile(self, bs):
        profiles = getattr(self, 'region_service_profiles', None)
        if profiles is None or len(profiles) <= bs:
            return np.ones(self.num_service_types, dtype=np.float64) / max(self.num_service_types, 1)
        return self._normalize_prob(profiles[bs])

    def _event_content_vector(self, bs):
        vec = np.zeros(self.num_content_F, dtype=np.float64)
        events = getattr(self, 'active_events', None)
        if events is None or bs >= len(events) or events[bs] is None:
            return vec
        center = int(events[bs].get('center', 0)) % max(self.num_content_F, 1)
        radius = max(2, self.num_content_F // 30)
        idx = (np.arange(center - radius, center + radius + 1) % self.num_content_F).astype(int)
        vec[idx] = 1.0
        return self._normalize_prob(vec)

    def _compose_content_profile(self, user, bs):
        global_pf = self._normalize_prob(getattr(user, 'global_profile', []))
        regional_pf = self._normalize_prob(getattr(user, 'regional_profile', []))
        if global_pf.size != self.num_content_F:
            global_pf = np.ones(self.num_content_F, dtype=np.float64) / max(self.num_content_F, 1)
        if regional_pf.size != self.num_content_F:
            regional_pf = global_pf.copy()

        phases = getattr(self, 'regional_phases', None)
        in_regional_phase = phases is not None and bs < len(phases) and bool(phases[bs])
        regional_strength = float(np.clip(getattr(self, 'regional_demand_strength', 1.0), 0.1, 5.0))
        active_regional_weight = float(np.clip(0.75 + 0.08 * (regional_strength - 1.0), 0.45, 0.95))
        inactive_regional_weight = float(np.clip(0.45 + 0.06 * (regional_strength - 1.0), 0.20, 0.85))
        regional_weight = active_regional_weight if in_regional_phase else inactive_regional_weight
        profile = regional_weight * regional_pf + (1.0 - regional_weight) * global_pf

        phase, _ = self._temporal_phase_index()
        temporal_shift = (phase + 1) * max(1, self.num_content_F // 17)
        temporal_pf = np.roll(regional_pf, temporal_shift)
        profile = 0.85 * profile + 0.15 * temporal_pf

        event_strength = float(np.clip(getattr(self, 'demand_event_strength', 0.35), 0.0, 0.95))
        own_event = self._event_content_vector(bs)
        if own_event.sum() > 0:
            profile = (1.0 - event_strength) * profile + event_strength * own_event

        adjacency = getattr(self, 'bs_adjacency', None)
        corr = float(np.clip(getattr(self, 'spatial_demand_correlation', 0.25), 0.0, 0.95))
        if adjacency is not None and bs < len(adjacency) and corr > 0:
            neighbor_vec = np.zeros(self.num_content_F, dtype=np.float64)
            neighbor_count = 0
            for j in range(self.num_agents):
                if j != bs and adjacency[bs, j] > 0:
                    ev = self._event_content_vector(j)
                    if ev.sum() > 0:
                        neighbor_vec += ev
                        neighbor_count += 1
            if neighbor_count > 0:
                neighbor_vec = self._normalize_prob(neighbor_vec)
                profile = (1.0 - corr) * profile + corr * neighbor_vec

        noise_ratio = float(np.clip(getattr(self, 'preference_noise', 0.03), 0.0, 1.0))
        if noise_ratio > 0:
            noise = self._normalize_prob(np.random.uniform(0.0, 1.0, self.num_content_F))
            profile = (1.0 - noise_ratio) * profile + noise_ratio * noise

        return self._normalize_prob(profile)

    def _compose_service_profile(self, user, bs):
        region_pf = self._normalize_prob(getattr(user, 'service_profile', self._region_service_profile(bs)))
        if region_pf.size != self.num_service_types:
            region_pf = self._region_service_profile(bs)

        phase, _ = self._temporal_phase_index()
        temporal_profiles = getattr(self, 'temporal_service_profiles', None)
        if temporal_profiles is not None and len(temporal_profiles) > phase:
            temporal_pf = self._normalize_prob(temporal_profiles[phase])
        else:
            temporal_pf = np.ones(self.num_service_types, dtype=np.float64) / max(self.num_service_types, 1)

        temporal_weight = float(np.clip(getattr(self, 'service_temporal_weight', 0.40), 0.0, 1.0))
        profile = (1.0 - temporal_weight) * region_pf + temporal_weight * temporal_pf

        phases = getattr(self, 'regional_phases', None)
        if phases is not None and bs < len(phases) and phases[bs]:
            boost = np.ones(self.num_service_types, dtype=np.float64)
            regional_strength = float(np.clip(getattr(self, 'regional_demand_strength', 1.0), 0.1, 5.0))
            service_boost = 1.20 + 0.08 * (regional_strength - 1.0)
            if self.num_service_types > 1:
                boost[1] *= service_boost
            if self.num_service_types > 2:
                boost[2] *= service_boost
            profile *= boost

        event_strength = float(np.clip(getattr(self, 'demand_event_strength', 0.35), 0.0, 0.95))
        events = getattr(self, 'active_events', None)
        if events is not None and bs < len(events) and events[bs] is not None:
            svc = int(events[bs].get('service_type', 0))
            event_pf = np.zeros(self.num_service_types, dtype=np.float64)
            event_pf[svc] = 1.0
            profile = (1.0 - event_strength) * profile + event_strength * event_pf

        return self._normalize_prob(profile)

    def _update_user_multimedia_demand(self, user):
        bs = int(getattr(user, 'associated_bs', 0))
        bs = int(np.clip(bs, 0, max(self.num_agents - 1, 0)))

        content_profile = self._compose_content_profile(user, bs)
        user.pop_profile_vector = content_profile.tolist()
        user.state.user_content_req = random_pick(user.content_rank, content_profile)

        service_profile = self._compose_service_profile(user, bs)
        user.current_service_profile = service_profile
        user.service_type = int(np.random.choice(self.num_service_types, p=service_profile))

    def _get_distance(self, agent, user):
        ap_loc = agent.location if agent.state.location_state is None else agent.state.location_state
        return np.sqrt((ap_loc[0] - user.location[0]) ** 2 + (ap_loc[1] - user.location[1]) ** 2)

    def update_request_history(self):
        if self.num_content_F is None or self.num_agents is None:
            return
        if self.request_history is None or len(self.request_history) != self.num_agents:
            self.request_history = [[] for _ in range(self.num_agents)]
        if self.service_history is None or len(self.service_history) != self.num_agents:
            self.service_history = [[] for _ in range(self.num_agents)]

        max_window = max(1, int(getattr(self, 'cache_request_window', 250)))
        for user in self.users:
            bs = int(np.clip(getattr(user, 'associated_bs', 0), 0, max(self.num_agents - 1, 0)))
            req = int(getattr(user.state, 'user_content_req', 0)) - 1
            if 0 <= req < self.num_content_F:
                hist = self.request_history[bs]
                hist.append(req)
                if len(hist) > max_window:
                    del hist[:-max_window]

            svc = int(getattr(user, 'service_type', 0))
            if 0 <= svc < self.num_service_types:
                svc_hist = self.service_history[bs]
                svc_hist.append(svc)
                if len(svc_hist) > max_window:
                    del svc_hist[:-max_window]

    def get_agent_popularity(self, agent):
        served_users = [
            user for user in self.users
            if getattr(user, 'associated_bs', agent.serial_number) == agent.serial_number
        ]
        if not served_users:
            served_users = self.users

        pop = np.zeros(self.num_content_F, dtype=np.float64)
        for user in served_users:
            user_pop = np.asarray(getattr(user, 'pop_profile_vector', []), dtype=np.float64)
            if len(user_pop) == self.num_content_F and np.isfinite(user_pop).all():
                pop += user_pop

        if self.request_history is not None and agent.serial_number < len(self.request_history):
            hist = self.request_history[agent.serial_number]
            if len(hist) > 0:
                recent = np.bincount(hist, minlength=self.num_content_F).astype(np.float64)
                if recent.sum() > 0:
                    recent /= recent.sum()
                    pop = 0.70 * pop + 0.30 * recent * max(len(served_users), 1)

        return self._normalize_prob(pop)

    def get_agent_service_mix(self, agent):
        mix = np.zeros(self.num_service_types, dtype=np.float64)
        if self.service_history is not None and agent.serial_number < len(self.service_history):
            hist = self.service_history[agent.serial_number]
            if len(hist) > 0:
                mix += np.bincount(hist, minlength=self.num_service_types).astype(np.float64)

        served_users = [
            user for user in self.users
            if getattr(user, 'associated_bs', agent.serial_number) == agent.serial_number
        ]
        for user in served_users:
            profile = np.asarray(getattr(user, 'current_service_profile', []), dtype=np.float64)
            if profile.size == self.num_service_types:
                mix += profile

        return self._normalize_prob(mix)

    def build_cache_state(self, agent, action_cache):
        capacity = max(0, min(int(agent.max_content_cached), self.num_content_F))
        new_cache = np.zeros(self.num_content_F, dtype=np.int32)
        if capacity <= 0:
            return new_cache

        pop = self.get_agent_popularity(agent)
        cache_score = pop.copy()
        if action_cache is not None:
            action_score = np.asarray(action_cache, dtype=np.float64).reshape(-1)
            if len(action_score) == self.num_content_F and np.isfinite(action_score).all():
                action_score = np.clip(action_score, 0.0, 1.0)
                if action_score.sum() > 0:
                    action_score = action_score / action_score.sum()
                    action_weight = float(np.clip(getattr(self, 'cache_policy_blend', 0.95), 0.0, 1.0))
                    cache_score = action_weight * action_score + (1.0 - action_weight) * pop
                    cache_score = self._normalize_prob(cache_score)

        total_steps = max(1, int(getattr(self, 'episode_counter', 0)))
        steps_per_episode = max(1, int(getattr(self, 'episode_steps_for_curriculum', 50)))
        episode = total_steps / float(steps_per_episode)
        curriculum_episodes = max(1.0, float(getattr(self, 'cache_curriculum_episodes', 200)))
        pop_ratio = min(1.0, episode / curriculum_episodes)

        n_pop = int(capacity * pop_ratio)
        n_rand = capacity - n_pop

        if n_pop > 0:
            keep_idx = np.argpartition(cache_score, -n_pop)[-n_pop:]
            new_cache[keep_idx] = 1

        if n_rand > 0:
            avail = np.where(new_cache == 0)[0]
            if len(avail) >= n_rand:
                rand_idx = np.random.choice(avail, n_rand, replace=False)
                new_cache[rand_idx] = 1

        return new_cache

    def update_user_pop_profile(self):
        if hasattr(self, 'regional_phases'):
            for i in range(len(self.regional_phases)):
                switch_prob = 0.10 + 0.20 * (i / max(1, len(self.regional_phases) - 1))
                if np.random.random() < switch_prob:
                    self.regional_phases[i] = not self.regional_phases[i]

        for user in self.users:
            user.preference_level = getattr(user, 'preference_level', 0) + 1
