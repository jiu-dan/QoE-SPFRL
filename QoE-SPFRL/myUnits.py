import random
import numpy as np
from sklearn import preprocessing
import math


def task_info(services, num_service_type):
    service_file = services
    task_size = np.zeros(num_service_type)

    delay_type1 = [0.5, 1.25, 2.5]
    delay_type2 = [1.25, 2.5, 3.75]
    delay_type3 = [2.5, 12.5, 20.0]

    T_max = np.zeros(num_service_type)
    T_min = np.zeros(num_service_type)
    T_fair = np.zeros(num_service_type)
    requested_cycles = np.zeros(num_service_type)

    task_size_p = [
        19.17022005, 22.20324493, 15.00114375, 18.02332573, 16.46755891,
        15.92338595, 16.86260211, 18.45560727, 18.96767474, 20.38816734,
        19.19194514, 21.85219500, 17.04452250, 23.78117436, 15.27387593,
        21.70467510, 19.17304802, 20.58689828, 16.40386939, 16.98101489
    ]

    for i, _ in enumerate(service_file):
        task_size[i] = float(task_size_p[i % len(task_size_p)])

        delay_type = random.randint(1, 3)
        if delay_type == 1:
            delay = delay_type1
        elif delay_type == 2:
            delay = delay_type2
        else:
            delay = delay_type3

        T_min[i] = delay[0]
        T_fair[i] = delay[1]
        T_max[i] = delay[2]
        requested_cycles[i] = 750.0

    Norm_task_size = norm_task_size(task_size)
    Norm_T_max = norm_T_max(T_max)
    Norm_T_min = norm_T_min(T_min)
    Norm_T_fair = norm_T_fair(T_fair)
    Norm_requested_cycles = norm_requested_cycles(requested_cycles)

    return (
        Norm_task_size, Norm_T_max, Norm_T_min, Norm_T_fair, Norm_requested_cycles,
        task_size, T_max, T_min, T_fair, requested_cycles
    )


def norm_task_size(task_size):
    mm = preprocessing.MinMaxScaler()
    Norm_task_size = mm.fit_transform(np.expand_dims(task_size, axis=1))
    return Norm_task_size

def norm_T_max(T_max):
    mm = preprocessing.MinMaxScaler()
    Norm_T_max = mm.fit_transform(np.expand_dims(T_max, axis=1))
    return Norm_T_max

def norm_T_min(T_min):
    mm = preprocessing.MinMaxScaler()
    Norm_T_min = mm.fit_transform(np.expand_dims(T_min, axis=1))
    return Norm_T_min

def norm_T_fair(T_fair):
    mm = preprocessing.MinMaxScaler()
    Norm_T_fair = mm.fit_transform(np.expand_dims(T_fair, axis=1))
    return Norm_T_fair

def norm_requested_cycles(requested_cycles):
    mm = preprocessing.MinMaxScaler()
    Norm_requested_cycles = mm.fit_transform(np.expand_dims(requested_cycles, axis=1))
    return Norm_requested_cycles


def random_pick(content_rank, probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return random.choices(content_rank, weights=probabilities, k=1)[0]


def get_gain(world, agent):
    distance = get_distance(world)
    gain = np.zeros(agent.num_UEs * world.num_agents)
    for i in range(world.num_agents):
        if i == agent.serial_number:
            Gain = np.random.uniform(world.radius/2, world.radius/2, [agent.num_UEs, 1]) ** world.gain_coefficient
            gain[agent.num_UEs * i: agent.num_UEs * i + agent.num_UEs] = Gain.squeeze()
        else:
            Gain = np.random.uniform(distance[agent.serial_number, i],
                                               distance[agent.serial_number, i],
                                               [agent.num_UEs, 1]) ** world.gain_coefficient
            gain[agent.num_UEs * i: agent.num_UEs * i + agent.num_UEs] = Gain.squeeze()

    mm = preprocessing.MinMaxScaler()
    norm_gain = mm.fit_transform(np.expand_dims(gain, axis=1))

    return gain, norm_gain.squeeze()


def get_distance(world):
    distance = np.ones([world.num_agents, world.num_agents])
    i = 0
    for _ in world.position:
        j = 0
        for _ in world.position:
            distance[i, j] = np.sqrt((world.position[i][0] - world.position[j][0])
                                     ** 2 + (world.position[i][1] - world.position[j][1]) ** 2)
            j = j + 1
        i = i + 1

    return distance


def get_adj(num_agents, positions=None, k=3, self_loop=True, fallback="full"):
    A = np.zeros((num_agents, num_agents), dtype=np.float32)

    if positions is not None and len(positions) == num_agents:
        positions = np.asarray(positions, dtype=np.float32)
        dist = np.linalg.norm(
            positions[:, None, :] - positions[None, :, :],
            axis=-1
        )

        kk = min(int(k), max(num_agents - 1, 0))
        for i in range(num_agents):
            if kk > 0:
                neighbor_idx = np.argsort(dist[i])[1:kk + 1]
                for j in neighbor_idx:
                    A[i, j] = 1.0
                    A[j, i] = 1.0

    else:
        if fallback == "ring":
            for i in range(num_agents):
                A[i, (i - 1) % num_agents] = 1.0
                A[i, (i + 1) % num_agents] = 1.0
        else:
            A[:] = 1.0

    if self_loop:
        np.fill_diagonal(A, 1.0)
    else:
        np.fill_diagonal(A, 0.0)

    return A
