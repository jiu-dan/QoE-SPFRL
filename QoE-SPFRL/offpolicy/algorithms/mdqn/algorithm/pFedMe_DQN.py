import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from offpolicy.algorithms.mdqn.algorithm.dqn import M_DQN
from offpolicy.utils.util import to_torch
import numpy as np


class pFedMe_DQN(nn.Module):
    def __init__(self, args, num_agents, cent_obs_dim, action_dim, device, mu=10, lr=1e-4, K=3):
        super(pFedMe_DQN, self).__init__()

        self.num_agents = num_agents
        self.device = device
        self.mu = mu
        self.lr = lr
        self.K = K
        self.args = args

        self.action_dims = list(getattr(args, "action_dims"))
        self.gamma = getattr(args, "gamma", 0.98)

        self.q_reduce = getattr(args, "q_reduce", "mean")
        self.reward_scale = getattr(args, "reward_scale", 1.0)
        self.target_clip = getattr(args, "target_clip", 50.0)

        self.beta = getattr(args, "beta", 0.5)
        self.sync_tau = getattr(args, "sync_tau", 0.005)
        self.target_update_tau = getattr(args, "target_update_tau", 0.005)
        self.max_grad_norm = getattr(args, "max_grad_norm", 1.0)
        self.lamda = getattr(args, "lamda", 0.3)

        self.p_fedavg_interval = max(1, int(getattr(args, "p_fedavg_interval", 5)))
        self.reg_warmup_episodes = max(0, int(getattr(args, "reg_warmup_episodes", 100)))
        self.advantage_warmup_episodes = max(1, int(getattr(args, "advantage_warmup_episodes", 250)))

        self.num_content_F = int(getattr(args, "num_content_F", 0))
        self.num_users = int(getattr(args, "num_users", 0))

        self.cache_capacities = list(
            getattr(args, "cache_capacities", [self.num_content_F] * num_agents)
        )

        self.global_model = M_DQN(args, num_agents, cent_obs_dim, action_dim, device).to(device)
        self.client_models = nn.ModuleList(
            [copy.deepcopy(self.global_model).to(device) for _ in range(num_agents)]
        )
        self.personal_models = nn.ModuleList(
            [copy.deepcopy(self.global_model).to(device) for _ in range(num_agents)]
        )

        self.personal_optimizers = [
            torch.optim.Adam(self.personal_models[i].parameters(), lr=self.lr, eps=1e-5)
            for i in range(num_agents)
        ]

        self.target_personal_models = nn.ModuleList(
            [copy.deepcopy(self.personal_models[i]).to(device) for i in range(num_agents)]
        )

        self.step_counter = 0
        self.batch_counter = 0
        self.current_episode = 0


        for model in self.target_personal_models:
            model.eval()

    def checkpoint_state_dict(self):
        return {
            key: value
            for key, value in self.state_dict().items()
            if key.startswith("global_model.") or key.startswith("personal_models.")
        }

    def load_checkpoint_state(self, state_dict):
        has_personal = any(key.startswith("personal_models.") for key in state_dict)
        incompatible = self.load_state_dict(state_dict, strict=False)

        with torch.no_grad():
            if not has_personal:
                global_state = self.global_model.state_dict()
                for model in self.personal_models:
                    model.load_state_dict(global_state)

            for personal, client, target in zip(
                self.personal_models, self.client_models, self.target_personal_models
            ):
                client.load_state_dict(personal.state_dict())
                target.load_state_dict(personal.state_dict())
                target.eval()

        return incompatible

    def _add_cache_popularity_bias(self, scores, agent_idx):
        pop_vector = self._get_popularity_vector(agent_idx)
        if pop_vector is None or len(pop_vector) != scores.size(1):
            return scores

        pop = torch.as_tensor(pop_vector, dtype=scores.dtype, device=scores.device).clamp_min(1e-12)
        pop = torch.log(pop) - torch.log(pop).mean()
        pop = pop / (pop.std(unbiased=False) + 1e-6)

        score_std = scores.detach().std(dim=1, keepdim=True, unbiased=False).clamp_min(0.05)
        weight = float(getattr(self.args, "cache_popularity_weight", 1.25))
        weight = weight * self._advantage_warmup_factor()
        return scores + weight * score_std * pop.unsqueeze(0)

    def _combine_head_q(self, q_list, reduce_mode=None):
        if len(q_list) == 0:
            return None

        reduce_mode = self.q_reduce if reduce_mode is None else reduce_mode
        stacked = torch.stack(q_list, dim=0)

        if reduce_mode == "sum":
            return stacked.sum(dim=0)
        return stacked.mean(dim=0)

    def _parameter_group(self, name):
        if "q_heads." not in name:
            return "backbone"
        try:
            head_idx = int(name.split("q_heads.", 1)[1].split(".", 1)[0])
        except (ValueError, IndexError):
            return "other"
        return "cache" if head_idx < self.num_content_F else "other"

    def _normalized_proximal_losses(self, personal_model, client_model):
        anchors = dict(client_model.named_parameters())
        sums = {
            "backbone": torch.zeros((), device=self.device),
            "cache": torch.zeros((), device=self.device),
            "other": torch.zeros((), device=self.device),
        }
        counts = {"backbone": 0, "cache": 0, "other": 0}

        for name, parameter in personal_model.named_parameters():
            group = self._parameter_group(name)
            anchor = anchors[name].detach()
            sums[group] = sums[group] + torch.sum((parameter - anchor) ** 2)
            counts[group] += parameter.numel()

        return tuple(
            sums[group] / max(1, counts[group])
            for group in ("backbone", "cache", "other")
        )

    def _last_step_tensor(self, x):
        if x.dim() == 3:
            return x[:, -1, :]
        if x.dim() == 2:
            return x
        if x.dim() == 1:
            return x.unsqueeze(-1)
        return x

    def _extract_agent_q(self, q_values, agent_idx):
        if isinstance(q_values, list):
            return [q[:, agent_idx, :] for q in q_values]
        if q_values.dim() == 2:
            return [q_values]
        if q_values.dim() == 3:
            return [q_values[:, agent_idx, :]]
        return q_values

    def set_current_episode(self, episode: int):
        self.current_episode = max(0, int(episode))

    def _advantage_warmup_factor(self) -> float:
        if self.advantage_warmup_episodes <= 0:
            return 1.0
        return min(1.0, max(0.0, float(self.current_episode) / float(self.advantage_warmup_episodes)))

    def train_policy_on_batch(self, batch_dict, use_same_share_obs=True):
        train_info = {}
        total_loss = 0.0

        for agent_idx in batch_dict.keys():
            loss = self.local_update(batch_dict[agent_idx], agent_idx)
            total_loss += float(loss)
            train_info[f"agent{agent_idx}_loss"] = float(loss)

        self.batch_counter += 1
        if self.batch_counter % self.p_fedavg_interval == 0:
            self.aggregate_global()
            self.broadcast_global()

        train_info["average_loss"] = total_loss / max(len(batch_dict), 1)
        return train_info, None, None

    def update_target_networks(self, tau=None):
        tau = self.target_update_tau if tau is None else tau

        with torch.no_grad():
            for online_model, target_model in zip(self.personal_models, self.target_personal_models):
                for p_tgt, p_online in zip(target_model.parameters(), online_model.parameters()):
                    p_tgt.data.mul_(1.0 - tau).add_(p_online.data, alpha=tau)

    def _build_capacity_constrained_next_q(self, q_next_online, q_next_target, agent_id):
        next_q_list = []

        cache_heads = min(self.num_content_F, len(q_next_online))
        capacity = int(self.cache_capacities[agent_id])
        capacity = max(0, min(capacity, cache_heads))

        if cache_heads > 0:
            score_list = []

            for h in range(cache_heads):
                q_on = q_next_online[h]

                if q_on.size(1) >= 2:
                    score = q_on[:, 1] - q_on[:, 0]
                else:
                    score = q_on[:, 0]

                score_list.append(score)

            scores = torch.stack(score_list, dim=1)
            scores = self._add_cache_popularity_bias(scores, agent_id)

            cache_actions = torch.zeros_like(scores, dtype=torch.long)

            if capacity > 0:
                topk_idx = torch.topk(scores, k=capacity, dim=1).indices
                cache_actions.scatter_(1, topk_idx, 1)

            for h in range(cache_heads):
                act_h = cache_actions[:, h].unsqueeze(1)
                act_h = act_h.clamp(max=q_next_target[h].size(1) - 1)
                next_q_list.append(q_next_target[h].gather(1, act_h))

        for h in range(cache_heads, len(q_next_online)):
            best_act = torch.argmax(q_next_online[h], dim=1, keepdim=True)
            next_q_list.append(q_next_target[h].gather(1, best_act))

        if not next_q_list:
            raise RuntimeError("No action heads available for target-Q construction")
        return self._combine_head_q(next_q_list)

    def local_update(self, sample, agent_id):
        client_model = self.client_models[agent_id]
        personal_model = self.personal_models[agent_id]
        target_model = self.target_personal_models[agent_id]
        optimizer = self.personal_optimizers[agent_id]

        personal_model.train()

        obs_batch = to_torch(sample["share_obs"]).float().to(self.device)
        next_obs_batch = to_torch(sample["next_share_obs"]).float().to(self.device)
        act_batch = to_torch(sample["actions"]).long().to(self.device)
        rew_batch = to_torch(sample["rewards"]).float().to(self.device)
        done_batch = to_torch(sample["dones"]).float().to(self.device)

        curr_actions = self._last_step_tensor(act_batch)
        curr_rewards = self._last_step_tensor(rew_batch)
        curr_dones = self._last_step_tensor(done_batch)
        curr_rewards = curr_rewards * float(self.reward_scale)

        with torch.no_grad():
            q_next_target_fixed = target_model.forward_agent(next_obs_batch, agent_id)

        warmup = float(self.reg_warmup_episodes)
        if warmup > 0 and self.current_episode < warmup:
            warmup_scale = float(self.current_episode) / warmup
        else:
            warmup_scale = 1.0

        loss_val = 0.0
        for _ in range(self.K):
            q_values = personal_model.forward_agent(obs_batch, agent_id)

            cache_q_list = []
            other_q_list = []
            for head_idx, q_head in enumerate(q_values):
                action = curr_actions[:, head_idx].unsqueeze(-1)
                action = action.clamp(max=q_head.size(1) - 1)
                q_val = q_head.gather(1, action)
                if head_idx < self.num_content_F:
                    cache_q_list.append(q_val)
                else:
                    other_q_list.append(q_val)

            chosen_heads = cache_q_list + other_q_list
            if not chosen_heads:
                raise RuntimeError("No action heads available for current-Q construction")
            chosen_q = self._combine_head_q(chosen_heads)

            with torch.no_grad():
                q_next_online = personal_model.forward_agent(next_obs_batch, agent_id)
                max_q_next = self._build_capacity_constrained_next_q(
                    q_next_online=q_next_online,
                    q_next_target=q_next_target_fixed,
                    agent_id=agent_id,
                )
                q_target = curr_rewards + self.gamma * max_q_next * (1.0 - curr_dones)
                q_target = torch.clamp(q_target, -self.target_clip, self.target_clip)

            td_loss = F.smooth_l1_loss(chosen_q, q_target)

            reg_backbone, reg_cache, reg_other_head = self._normalized_proximal_losses(
                personal_model, client_model
            )

            mu_backbone = warmup_scale * self.mu * 0.3
            mu_cache = warmup_scale * self.mu * 0.03
            mu_other_head = warmup_scale * self.mu * 0.5

            reg_loss = 0.5 * (
                mu_backbone * reg_backbone
                + mu_cache * reg_cache
                + mu_other_head * reg_other_head
            )

            total_loss = td_loss + reg_loss

            self.step_counter += 1

            if agent_id == 0 and self.step_counter % 100 == 0:
                ratio = reg_loss / (td_loss + 1e-8)
                print(f"[Step {self.step_counter}] [Agent {agent_id}] TD Loss: {td_loss.item():.6f} | "
                      f"Reg Loss: {reg_loss.item():.6f} | "
                      f"Ratio: {ratio.item():.4f} | "
                      f"Warmup: {warmup_scale:.2f}")

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(personal_model.parameters(), self.max_grad_norm)
            optimizer.step()

            loss_val = total_loss.item()

        with torch.no_grad():
            alpha = self.lamda
            for p_client, p_theta in zip(client_model.parameters(), personal_model.parameters()):
                p_client.data.copy_((1.0 - alpha) * p_client.data + alpha * p_theta.data)

        return loss_val

    def aggregate_global(self):
        with torch.no_grad():
            global_dict = self.global_model.state_dict()
            avg_dict = copy.deepcopy(global_dict)

            for k in avg_dict.keys():
                avg_dict[k] = torch.zeros_like(avg_dict[k])

            for client_model in self.client_models:
                local_dict = client_model.state_dict()
                for k in avg_dict.keys():
                    avg_dict[k] += local_dict[k] / self.num_agents

            for k in global_dict.keys():
                global_dict[k] = (1.0 - self.beta) * global_dict[k] + self.beta * avg_dict[k]

            self.global_model.load_state_dict(global_dict)

    def broadcast_global(self, tau=None):
        tau = self.sync_tau if tau is None else tau
        if tau <= 0.0:
            return
        with torch.no_grad():
            for client_model in self.client_models:
                for p_client, p_global in zip(client_model.parameters(), self.global_model.parameters()):
                    p_client.data.copy_((1.0 - tau) * p_client.data + tau * p_global.data)


    def select_action(self, agent_idx, states, epsilon=0.0):
        if not torch.is_tensor(states):
            states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        else:
            states = states.to(self.device, dtype=torch.float32)
        if states.dim() == 1:
            states = states.unsqueeze(0)

        batch_size = states.shape[0]
        self.personal_models[agent_idx].eval()

        with torch.no_grad():
            q_heads = self.get_agent_q(agent_idx, states)

        num_heads = len(q_heads)
        cache_heads = min(self.num_content_F, num_heads)
        capacity = int(self.cache_capacities[agent_idx])
        capacity = max(0, min(capacity, cache_heads))

        random_mask = torch.rand(batch_size, device=self.device) < epsilon
        cache_epsilon = epsilon * float(getattr(self.args, "cache_explore_epsilon_scale", 0.25)) * self._advantage_warmup_factor()
        cache_random_mask = torch.rand(batch_size, device=self.device) < cache_epsilon

        if cache_heads > 0:
            scores = torch.stack(
                [q_heads[h][:, 1] - q_heads[h][:, 0] for h in range(cache_heads)], dim=1
            )
            scores = self._add_cache_popularity_bias(scores, agent_idx)

            if capacity > 0:
                _, topk_idx = torch.topk(scores, k=capacity, dim=1)
                greedy_actions = torch.zeros(
                    batch_size, cache_heads, dtype=torch.long, device=self.device
                )
                greedy_actions.scatter_(1, topk_idx, 1)
            else:
                greedy_actions = torch.zeros(
                    batch_size, cache_heads, dtype=torch.long, device=self.device
                )

            if cache_epsilon > 0 and cache_random_mask.any():
                random_actions = torch.zeros(
                    batch_size, cache_heads, dtype=torch.long, device=self.device
                )
                explore_idx = torch.where(cache_random_mask)[0].cpu().numpy()
                pop_vector = self._get_popularity_vector(agent_idx)
                if pop_vector is not None and len(pop_vector) == cache_heads:
                    pop_power = float(getattr(self.args, "cache_explore_pop_power", 0.7))
                    pop_power = pop_power * self._advantage_warmup_factor()
                    if pop_power <= 0.0:
                        pop_prob = None
                    else:
                        pop_prob = np.power(np.clip(pop_vector, 1e-12, None), pop_power)
                        pop_prob /= pop_prob.sum()
                else:
                    pop_prob = None

                for b in explore_idx:
                    if pop_prob is not None:
                        chosen = np.random.choice(cache_heads, size=capacity, replace=False, p=pop_prob)
                    else:
                        chosen = np.random.choice(cache_heads, size=capacity, replace=False)
                    chosen_idx = torch.as_tensor(chosen, dtype=torch.long, device=self.device)
                    random_actions[b, chosen_idx] = 1

                cache_actions = torch.where(cache_random_mask.unsqueeze(1), random_actions, greedy_actions)
            else:
                cache_actions = greedy_actions
        else:
            cache_actions = None

        other_actions = []
        for h in range(cache_heads, num_heads):
            q_vals = q_heads[h]
            greedy_act = torch.argmax(q_vals, dim=1)
            random_act = torch.randint(
                0, q_vals.shape[1], (batch_size,), device=self.device
            )
            act = torch.where(random_mask, random_act, greedy_act)
            other_actions.append(act.unsqueeze(1))

        if cache_heads > 0:
            all_actions = [cache_actions] + other_actions
        else:
            all_actions = other_actions

        actions = torch.cat(all_actions, dim=1).cpu().numpy()
        actions = self._apply_qoe_action_prior(actions, agent_idx, epsilon)
        return actions

    def get_agent_q(self, agent_idx, states):
        if not torch.is_tensor(states):
            states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        else:
            states = states.to(self.device, dtype=torch.float32)

        return self.personal_models[agent_idx].forward_agent(states, agent_idx)

    def _get_popularity_vector(self, agent_idx):
        try:
            world = getattr(self, 'world', None)
            if world is None:
                return None
            agent = world.agents[agent_idx]
            served_users = [
                u for u in world.users
                if getattr(u, 'associated_bs', agent.serial_number) == agent.serial_number
            ]
            if len(served_users) == 0:
                return None
            freq = np.zeros(self.num_content_F, dtype=np.float64)
            for u in served_users:
                pop = np.array(u.pop_profile_vector, dtype=np.float64)
                freq += pop
            total = freq.sum()
            if not np.isfinite(total) or total <= 0:
                return None
            freq = freq / total
            return freq
        except Exception:
            return None

    def _apply_qoe_action_prior(self, actions, agent_idx, epsilon):
        if not bool(getattr(self.args, "use_qoe_action_prior", True)):
            return actions

        warmup = self._advantage_warmup_factor()
        if warmup < 1.0 and np.random.random() > warmup:
            return actions

        world = getattr(self, "world", None)
        if world is None or self.num_users <= 0:
            return actions

        expected_dim = self.num_content_F + 3 * self.num_users
        if actions.shape[1] < expected_dim:
            return actions

        prior_explore_scale = float(getattr(self.args, "qoe_action_prior_explore_scale", 0.25))
        skip_prob = float(np.clip(epsilon * prior_explore_scale, 0.0, 1.0))
        action_dims = self.action_dims

        offload_offset = self.num_content_F
        compute_offset = offload_offset + self.num_users
        bandwidth_offset = compute_offset + self.num_users

        for b in range(actions.shape[0]):
            if skip_prob > 0.0 and np.random.random() < skip_prob:
                continue

            for user in getattr(world, "users", []):
                user_id = int(getattr(user, "serial_number", -1))
                if user_id < 0 or user_id >= self.num_users:
                    continue

                if int(getattr(user, "associated_bs", agent_idx)) != int(agent_idx):
                    continue

                offload_idx = offload_offset + user_id
                compute_idx = compute_offset + user_id
                bandwidth_idx = bandwidth_offset + user_id

                if offload_idx < len(action_dims):
                    actions[b, offload_idx] = min(1, action_dims[offload_idx] - 1)
                if compute_idx < len(action_dims):
                    actions[b, compute_idx] = max(0, action_dims[compute_idx] - 1)
                if bandwidth_idx < len(action_dims):
                    actions[b, bandwidth_idx] = max(0, action_dims[bandwidth_idx] - 1)

        return actions


QoE_SPFRL = pFedMe_DQN
