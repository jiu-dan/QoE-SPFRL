import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from offpolicy.utils.util import to_torch
from myUnits import get_adj, get_gain


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a_l = nn.Parameter(torch.zeros(size=(out_features, 1)))
        self.a_l = nn.Parameter(torch.zeros(size=(out_features, 1)))
        self.a_r = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_l.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_r.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        h = torch.matmul(inp, self.W)

        score_l = torch.matmul(h, self.a_l)
        score_r = torch.matmul(h, self.a_r)
        e = score_l + score_r.transpose(1, 2)
        e = self.leakyrelu(e)

        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout, alpha, n_heads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = nn.ModuleList([
            GraphAttentionLayer(n_feat, n_hid, dropout, alpha, concat=True)
            for _ in range(n_heads)
        ])

        self.out_att = GraphAttentionLayer(
            n_hid * n_heads, n_class, dropout, alpha, concat=False
        )

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        return x


class RelationLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.05, alpha=0.2, n_heads=4):
        super(RelationLayer, self).__init__()

        self.gat = GAT(
            n_feat=in_dim,
            n_hid=hid_dim,
            n_class=out_dim,
            dropout=dropout,
            alpha=alpha,
            n_heads=n_heads
        )

        self.proj = nn.Linear(out_dim, out_dim)

        if in_dim == out_dim:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Linear(in_dim, out_dim)

        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        y = self.gat(x, adj)
        y = self.proj(y)
        y = y + self.skip(x)
        y = self.norm(y)
        y = F.relu(y)
        y = self.dropout(y)
        return y


class DuelingQHead(nn.Module):

    def __init__(self, input_dim, action_dim, hidden_dim=64, dropout=0.05):
        super(DuelingQHead, self).__init__()

        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        self.value = nn.Linear(hidden_dim, 1)
        self.advantage = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        z = self.feature(x)
        v = self.value(z)
        a = self.advantage(z)
        q = v + a - a.mean(dim=-1, keepdim=True)
        return q


class M_DQN(nn.Module):
    def __init__(self, args, num_agents, cent_obs_dim, action_dim, device):
        super(M_DQN, self).__init__()

        self.args = args
        self.num_agents = num_agents
        self.device = device

        self.action_dims = list(getattr(args, "action_dims"))
        self.num_action_heads = len(self.action_dims)
        self.seq_len = int(getattr(args, "seq_len", 8))

        if cent_obs_dim % num_agents != 0:
            raise ValueError(
                f"cent_obs_dim={cent_obs_dim} cannot be divided by num_agents={num_agents}"
            )

        self.n_feat = int(cent_obs_dim / num_agents)

        positions = getattr(args, "bs_positions", None)
        gat_k = int(getattr(args, "gat_k", 3))
        adj_fallback = getattr(args, "adj_fallback", "full")

        adj_np = get_adj(
            num_agents=num_agents,
            positions=positions,
            k=gat_k,
            self_loop=True,
            fallback=adj_fallback
        )

        self.adj = adj_np
        self.register_buffer(
            "adj_tensor",
            torch.as_tensor(adj_np, dtype=torch.float32)
        )

        encoder_dropout = float(getattr(args, "encoder_dropout", 0.05))
        gat_dropout = float(getattr(args, "gat_dropout", 0.05))
        gat_heads = int(getattr(args, "gat_heads", 4))

        hidden_dim = int(getattr(args, "dqn_hidden_dim", 64))
        lstm_hidden_dim = int(getattr(args, "lstm_hidden_dim", 64))
        q_hidden_dim = int(getattr(args, "q_hidden_dim", 64))

        self.obs_encoder = nn.Sequential(
            nn.Linear(self.n_feat, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(encoder_dropout),

            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        self.relation = RelationLayer(
            in_dim=hidden_dim,
            hid_dim=max(16, hidden_dim // 2),
            out_dim=hidden_dim,
            dropout=gat_dropout,
            alpha=0.2,
            n_heads=gat_heads
        )

        self.use_lstm = bool(getattr(args, "use_lstm", True))
        if self.use_lstm:
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=lstm_hidden_dim,
                num_layers=1,
                batch_first=True
            )
            self.temporal_norm = nn.LayerNorm(lstm_hidden_dim)
            self.q_input_dim = lstm_hidden_dim
        else:
            self.lstm = None
            self.temporal_norm = nn.LayerNorm(hidden_dim)
            self.q_input_dim = hidden_dim

        self.q_heads = nn.ModuleList([
            DuelingQHead(
                input_dim=self.q_input_dim,
                action_dim=dim,
                hidden_dim=q_hidden_dim,
                dropout=0.05
            )
            for dim in self.action_dims
        ])

    def _build_adj_batch(self, batch_size):
        adj = self.adj_tensor.to(self.device)
        return adj.unsqueeze(0).expand(batch_size, -1, -1)

    def _to_tensor(self, x):
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        elif not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        else:
            x = x.float().to(self.device)
        return x

    def _encode_spatial_per_timestep(self, states_seq):
        states_seq = self._to_tensor(states_seq)
        B, T, D = states_seq.shape
        expected_D = self.num_agents * self.n_feat
        if D != expected_D:
            raise ValueError(
                f"Expected state dim={expected_D}, but got D={D}. "
                f"num_agents={self.num_agents}, n_feat={self.n_feat}"
            )

        node_seq = states_seq.view(B, T, self.num_agents, self.n_feat)

        x_flat = node_seq.reshape(B * T, self.num_agents, self.n_feat)
        x_flat = self.obs_encoder(x_flat)

        adj_base = self._build_adj_batch(B)
        adj_flat = adj_base.unsqueeze(1).expand(B, T, -1, -1).reshape(B * T, self.num_agents, -1)

        s_flat = self.relation(x_flat, adj_flat)

        spatial_seq = s_flat.view(B, T, self.num_agents, -1).permute(0, 2, 1, 3)
        return spatial_seq

    def encode(self, states):
        states = self._to_tensor(states)
        if states.dim() == 1:
            states = states.unsqueeze(0)
        if states.dim() == 2:
            states = states.unsqueeze(1)
        if states.dim() != 3:
            raise ValueError(
                f"Expected states shape [B, D] or [B, T, D], got {states.shape}"
            )

        B, T, D = states.shape

        spatial_seq = self._encode_spatial_per_timestep(states)

        if self.use_lstm and self.lstm is not None:
            lstm_in = spatial_seq.reshape(B * self.num_agents, T, -1)
            lstm_out, _ = self.lstm(lstm_in)
            last_hidden = lstm_out[:, -1, :]
            last_hidden = self.temporal_norm(last_hidden)
            x = last_hidden.view(B, self.num_agents, -1)
        else:
            pooled = spatial_seq.mean(dim=2)
            pooled = self.temporal_norm(pooled)
            x = pooled

        return x

    def forward(self, states):
        x = self.encode(states)
        return [head(x) for head in self.q_heads]

    def forward_agent(self, states, agent_idx):
        if agent_idx < 0 or agent_idx >= self.num_agents:
            raise IndexError(f"agent_idx={agent_idx} outside [0, {self.num_agents})")
        x = self.encode(states)[:, agent_idx, :]
        return [head(x) for head in self.q_heads]
