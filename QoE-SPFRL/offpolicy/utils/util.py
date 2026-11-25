import copy
import random

import gym
import numpy as np
from gym.spaces import Box, Discrete, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Variable
from myUnits import random_pick
def to_torch(input):
    return torch.from_numpy(input) if type(input) == np.ndarray else input

def to_numpy(x):
    return x.detach().cpu().numpy()

class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample()

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class MultiDiscrete(gym.Space):

    def __init__(self, array_of_param_array):
        self.low = np.array([x[0] for x in array_of_param_array])
        self.high = np.array([x[1] for x in array_of_param_array])
        self.num_discrete_space = self.low.shape[0]
        self.n = np.sum(self.high) + 2

    def sample(self):
        random_array = np.random.rand(self.num_discrete_space)
        return [int(x) for x in np.floor(np.multiply((self.high - self.low + 1.), random_array) + self.low)]

    def contains(self, x):
        return len(x) == self.num_discrete_space and (np.array(x) >= self.low).all() and (np.array(x) <= self.high).all()

    @property
    def shape(self):
        return self.num_discrete_space

    def __repr__(self):
        return "MultiDiscrete" + str(self.num_discrete_space)

    def __eq__(self, other):
        return np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high)


class DecayThenFlatSchedule():
    def __init__(self,
                 start,
                 finish,
                 time_length,
                 decay="exp"):

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay

        if self.decay in ["exp"]:
            self.exp_scaling = (-1) * self.time_length / \
                np.log(self.finish) if self.finish > 0 else 1

    def eval(self, T):
        if self.decay in ["linear"]:
            return max(self.finish, self.start - self.delta * T)
        elif self.decay in ["exp"]:
            return min(self.start, max(self.finish, np.exp(- T / self.exp_scaling)))
    pass


def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)


def mse_loss(e):
    return e**2


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size


def onehot_from_logits(logits, avail_logits=None, eps=0.0):
    logits = to_torch(logits)
    
    dim = len(logits.shape) - 1
    if avail_logits is not None:
        avail_logits = to_torch(avail_logits)
        logits[avail_logits == 0] = -1e10
    argmax_acs = (logits == logits.max(dim, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])


def onehot_from_logits_cahnnel(logits, avail_logits=None, eps=0.0):
    logits = to_torch(logits)

    dim = len(logits.shape) - 1
    if avail_logits is not None:
        avail_logits = to_torch(avail_logits)
        logits[avail_logits == 0] = -1e10
    argmax_acs = (logits == logits.max(dim, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(range(logits.shape[1]), size=logits.shape[0])]],
                        requires_grad=False)
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])

def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, avail_logits, temperature, device=torch.device('cpu')):
    if str(device) == 'cpu':
        y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data))
    else:
        y = (logits.cpu() + sample_gumbel(logits.shape,
                                          tens_type=type(logits.data))).cuda()

    dim = len(logits.shape) - 1
    if avail_logits is not None:
        avail_logits = to_torch(avail_logits).to(device)
        y[avail_logits==0] = -1e10
    return F.softmax(y / temperature, dim=dim)

def gumbel_softmax(logits, max_storage, agent_id, avail_logits=None, temperature=1.0, hard=False, device=torch.device('cpu'), channel_constraint=None):
    rev_rank_1 = [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    rev_rank_2 = [20, 19, 18, 17, 16, 15, 14, 13, 6, 5, 4, 3, 2, 1, 12, 11, 10, 9, 8, 7]
    rev_rank_3 = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]
    rev_rank_4 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    if agent_id == 0 or agent_id == 1 or agent_id == 2:
        rev_rank = rev_rank_1
    elif agent_id == 4:
        rev_rank = rev_rank_2
    elif agent_id == 5 or agent_id == 6:
        rev_rank = rev_rank_3
    else:
        rev_rank = rev_rank_4
    required_storage = [15, 13, 14, 16, 15, 12, 10, 13, 16, 12, 17, 11, 12, 18, 12, 16, 15, 19, 12, 10]

    if len(logits) == 5:
        channel_constraint = []
        index_batch = []

        with torch.no_grad():
            for i in range(len(logits)):
                y = gumbel_softmax_sample(logits[i], avail_logits, temperature, device)
                channel_constraint.append(y)
                y_hard = onehot_from_logits(y)
                new_y_hard = y_hard.cpu().numpy().tolist()
                index_batch.append(list(map(lambda x: x.index(1.0), new_y_hard)))

        for batch in range(len(index_batch[0])):
            index = []
            for UE in range(len(index_batch)):
                index.append(index_batch[UE][batch])
            while not len(index) == len(set(index)):

                for i in range(len(index_batch)):
                    for j in range(i+1, len(index_batch)):
                        if index[i] == index[j]:
                            max_act = torch.argmax(channel_constraint[j][batch], dim=0, keepdim=False).tolist()
                            channel_constraint[j][batch] = torch.where(channel_constraint[j][batch] == channel_constraint[j][batch][max_act],
                                            channel_constraint[j][batch]/100,
                                            channel_constraint[j][batch])

                with torch.no_grad():
                    for i in range(len(index_batch)):
                        y_hard = onehot_from_logits(channel_constraint[i][batch])
                        new_y_hard = y_hard.cpu().numpy().tolist()
                        index[i] = new_y_hard.index(1.0)
        act_outs = []
        for i in range(len(logits)):
            y_hard = onehot_from_logits(channel_constraint[i])
            y = (y_hard - channel_constraint[i]).detach() + channel_constraint[i]
            act_outs.append(y)
        y = torch.cat(act_outs, dim=-1)

    elif len(logits) == 20:
        caching_constraint = []
        index_batch = []

        with torch.no_grad():
            for i in range(len(logits)):
                y = gumbel_softmax_sample(logits[i], avail_logits, temperature, device)
                caching_constraint.append(y)
                y_hard = onehot_from_logits(y)
                new_y_hard = y_hard.cpu().numpy().tolist()
                index_batch.append(list(map(lambda x: x.index(1.0), new_y_hard)))
        for batch in range(len(index_batch[0])):
            index = []
            for service in range(len(index_batch)):
                index.append(index_batch[service][batch])

            total_storage = sum(np.multiply(index, required_storage))
            while not total_storage <= max_storage:
                drop_service = 19
                for i in rev_rank:
                    if index[i - 1] == 1:
                        caching_constraint[i - 1][batch][1] = -1e10
                        drop_service = i - 1
                        break

                with torch.no_grad():
                    y_hard = onehot_from_logits(caching_constraint[drop_service][batch])
                    new_y_hard = y_hard.cpu().numpy().tolist()
                    index[drop_service] = new_y_hard.index(1.0)
                    total_storage = sum(np.multiply(index, required_storage))

        act_outs = []
        for i in range(len(logits)):
            y_hard = onehot_from_logits(caching_constraint[i])
            y = (y_hard - caching_constraint[i]).detach() + caching_constraint[i]
            act_outs.append(y)
        y = torch.cat(act_outs, dim=-1)

    else:
        y = gumbel_softmax_sample(logits, avail_logits, temperature, device)
        if hard:
            y_hard = onehot_from_logits(y)
            y = (y_hard - y).detach() + y

    return y


def gaussian_noise(shape, std):
    return torch.empty(shape).normal_(mean=0, std=std)


def get_obs_shape(obs_space):
    if obs_space.__class__.__name__ == "Box":
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == "list":
        obs_shape = obs_space
    else:
        raise NotImplementedError
    
    return obs_shape


def get_dim_from_space(space):
    if isinstance(space, Box):
        dim = int(np.prod(space.shape))

    elif isinstance(space, Discrete):
        dim = space.n

    elif isinstance(space, Tuple):
        dim = sum([get_dim_from_space(sp) for sp in space])

    elif isinstance(space, gym.spaces.MultiDiscrete):
        dim = int(np.sum(space.nvec))

    elif "MultiDiscrete" in space.__class__.__name__:
        if hasattr(space, "nvec"):
            dim = int(np.sum(space.nvec))
        elif hasattr(space, "high") and hasattr(space, "low"):
            dim = int(np.sum((space.high - space.low) + 1))
        else:
            raise Exception("Unsupported MultiDiscrete format")

    elif isinstance(space, list):
        dim = space[0]

    else:
        raise Exception("Unrecognized space: ", type(space))

    return dim


def get_state_dim(observation_dict, action_dict):
    combined_obs_dim = sum([get_dim_from_space(space)
                            for space in observation_dict.values()])
    combined_act_dim = 0
    for space in action_dict.values():
        dim = get_dim_from_space(space)
        if isinstance(dim, np.ndarray):
            combined_act_dim += int(sum(dim))
        else:
            combined_act_dim += dim
    return combined_obs_dim, combined_act_dim, combined_obs_dim+combined_act_dim


def get_cent_act_dim(action_space):
    cent_act_dim = 0
    for space in action_space:
        dim = get_dim_from_space(space)
        if isinstance(dim, np.ndarray):
            cent_act_dim += int(sum(dim))
        else:
            cent_act_dim += dim
    return cent_act_dim


def is_discrete(space):
    if isinstance(space, Discrete) or "MultiDiscrete" in space.__class__.__name__:
        return True
    else:
        return False


def is_multidiscrete(space):
    if "MultiDiscrete" in space.__class__.__name__:
        return True
    else:
        return False


def make_onehot(int_action, action_dim, seq_len=None):
    if type(int_action) == torch.Tensor:
        int_action = int_action.cpu().numpy()
    if not seq_len:
        return np.eye(action_dim)[int_action]
    if seq_len:
        onehot_actions = []
        for i in range(seq_len):
            onehot_action = np.eye(action_dim)[int_action[i]]
            onehot_actions.append(onehot_action)
        return np.stack(onehot_actions)


def avail_choose(x, avail_x=None):
    x = to_torch(x)
    if avail_x is not None:
        avail_x = to_torch(avail_x)
        x[avail_x == 0] = -1e10
    return x







def tile_images(img_nhwc):
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    img_nhwc = np.array(
        list(img_nhwc) + [img_nhwc[0]*0 for _ in range(N, H*W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
    return img_Hh_Ww_c

def avail_caching_choose(caching_act_out, max_content, num_content):
    pre_content_cached = []

    j = 0


    for i in range(num_content):
        if caching_act_out[j] < caching_act_out[j+1]:
            pre_content_cached.append(i)
        j = j + 2
    if len(pre_content_cached) > max_content:
        change = pre_content_cached[max_content::]

        for _, index in enumerate(change):
            caching_act_out[index*2] = 1
            caching_act_out[index * 2 + 1] = 0
    elif len(pre_content_cached) < max_content:
        remaining_content = [i for i in range(num_content) if i not in pre_content_cached]
        additional_content = remaining_content[:max_content - len(pre_content_cached)]
        for index in additional_content:
            caching_act_out[index * 2] = 0
            caching_act_out[index * 2 + 1] = 1

    return caching_act_out

def avail_power_choose(power_out, power_seg, num_user, max_power,  min_power, user_assoc_out):
    AP_user_assoc = []
    AP_user_assoc_power = []
    j = 0
    jj = 0
    for i in range(num_user):
        if user_assoc_out[j] < user_assoc_out[j+1]:
            AP_user_assoc.append(i)
            max_index = np.argmax(power_out[jj*power_seg:jj*power_seg + power_seg])
            AP_user_assoc_power.append(max_index)
        else:
            max_index = np.argmax(power_out[jj * power_seg:jj * power_seg + power_seg])
            power_out[jj*power_seg + max_index] = 0
            power_out[jj*power_seg] = 1
        j = j+2
        jj = jj + 1

    real_power_allocation = []
    for j in range(len(AP_user_assoc)):
        power_real = (((AP_user_assoc_power[j] - 0)
                            * (max_power - min_power)) /
                                (power_seg)) + min_power
        real_power_allocation.append(power_real)

    while sum(real_power_allocation) > max_power:
        max_value = np.max(AP_user_assoc_power)
        max_index = AP_user_assoc_power.index(max_value)
        index_tmp = AP_user_assoc[max_index]
        power_out[index_tmp*power_seg + max_value] = 0
        power_out[index_tmp * power_seg + max_value - 1] = 1

        AP_user_assoc = []
        AP_user_assoc_power = []
        j = 0
        jj = 0
        for i in range(num_user):
            if user_assoc_out[j] < user_assoc_out[j + 1]:
                AP_user_assoc.append(i)
                max_index = np.argmax(power_out[jj * power_seg:jj * power_seg + power_seg])
                AP_user_assoc_power.append(max_index)
            else:
                max_index = np.argmax(power_out[jj * power_seg:jj * power_seg + power_seg])
                power_out[jj * power_seg + max_index] = 0
                power_out[jj * power_seg] = 1
            j = j + 2
            jj = jj + 1

        real_power_allocation = []
        for j in range(len(AP_user_assoc)):
            power_real = (((AP_user_assoc_power[j] - 0)
                           * (max_power - min_power)) /
                          (power_seg)) + min_power
            real_power_allocation.append(power_real)

    return power_out



def avail_caching_choose_onebatch(caching_act_Q_out, max_storage, agent_id):
    rev_rank_1 = [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    rev_rank_2 = [20, 19, 18, 17, 16, 15, 14, 13, 6, 5, 4, 3, 2, 1, 12, 11, 10, 9, 8, 7]
    rev_rank_3 = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]
    rev_rank_4 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    if agent_id == 0 or agent_id == 1 or agent_id == 2:
        rev_rank = rev_rank_1
    elif agent_id == 4:
        rev_rank = rev_rank_2
    elif agent_id == 5 or agent_id == 6:
        rev_rank = rev_rank_3
    else:
        rev_rank = rev_rank_4
    required_storage = [15, 13, 14, 16, 15, 12, 10, 13, 16, 12, 17, 11, 12, 18, 12, 16, 15, 19, 12, 10]
    index = []
    for i in range(len(caching_act_Q_out)):
        caching_act_Q_out[i] = caching_act_Q_out[i].clone()
        greedy_action = (caching_act_Q_out[i].max(dim=-1))[1][0].tolist()
        index.append(greedy_action)

    total_storage = sum(np.multiply(index, required_storage))
    while not total_storage <= max_storage:
        drop_service = 19
        for i in rev_rank:
            if index[i-1] == 1:
                caching_act_Q_out[i-1][0][1] = -1e10
                drop_service = i-1
                break
        index[drop_service] = (caching_act_Q_out[drop_service].max(dim=-1))[1][0].tolist()

        total_storage = sum(np.multiply(index, required_storage))

    return caching_act_Q_out


def avail_caching_choose_random(caching_act_Q_out, max_storage, agent_id):
    rev_rank_1 = [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    rev_rank_2 = [20, 19, 18, 17, 16, 15, 14, 13, 6, 5, 4, 3, 2, 1, 12, 11, 10, 9, 8, 7]
    rev_rank_3 = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]
    rev_rank_4 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    if agent_id == 0 or agent_id == 1 or agent_id == 2:
        rev_rank = rev_rank_1
    elif agent_id == 4:
        rev_rank = rev_rank_2
    elif agent_id == 5 or agent_id == 6:
        rev_rank = rev_rank_3
    else:
        rev_rank = rev_rank_4
    required_storage = [15, 13, 14, 16, 15, 12, 10, 13, 16, 12, 17, 11, 12, 18, 12, 16, 15, 19, 12, 10]
    index = []
    for i in range(len(caching_act_Q_out)):
        act = np.where(caching_act_Q_out[i][0] == 1)
        index.append(act[0][0])

    total_storage = sum(np.multiply(index, required_storage))
    while not total_storage <= max_storage:
        drop_service = 19
        for i in rev_rank:
            if index[i-1] == 1:
                caching_act_Q_out[i-1][0][1] = 0
                caching_act_Q_out[i-1][0][0] = 1
                drop_service = i-1
                break
        index[drop_service] = np.where(caching_act_Q_out[drop_service][0] == 1)[0][0]
        total_storage = sum(np.multiply(index, required_storage))

    return caching_act_Q_out