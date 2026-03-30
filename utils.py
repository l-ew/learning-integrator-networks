import os
import csv
import numpy as np
from operator import itemgetter


def network_params_str(params):
    N, J0, b, sigma, r, theta_grid, input_structure, nonlin, loc_input, lmbda, online  = itemgetter('N', 'J0', 'b', 'sigma', 'r', 'theta_grid', 'input_structure', 'nonlinearity', 'loc_input', 'lmbda', 'online')(params)

    theta_str = 'grid' if theta_grid else 'random'

    input_map = {
        'double_ring': 'doublering',
        'three_ring': 'threering',
        'multi_ring': 'multiring',
        'random': 'random',
    }
    input_str = input_map[input_structure]

    network_params = 'N={:d}_J0={:.0f}_b={:.1f}_sigma={:.0f}_r={:.0f}_lambda={:.0f}_theta={}_input={}'.format(
        N, J0, b, sigma, r, lmbda, theta_str, input_str)

    network_params = network_params + '_fb=' + str(loc_input) + '_nonlin=' + str(nonlin) + '_online=' + str(online)

    return network_params


def get_data_path(path, params):
    network_params = network_params_str(params)
    return os.path.join(path, network_params)


def nonlin_str2fun(nonlin_str):
    if nonlin_str == 'exp':
        return np.exp
    elif nonlin_str == 'relu':
        return lambda x: x * (x > 0)
    elif nonlin_str == 'explin':
        return lambda x: np.log(1 + np.exp(x))
    elif nonlin_str == 'tanh':
        return lambda x: 1.0 + np.tanh(x)
    elif nonlin_str == 'heaviside':
        return lambda x: (x > 0).astype(float) + (x > 2).astype(float)
    else:
        print("Nonlinearity not implemented")


def loc_input_str2fun(loc_str, kappa=3, custom_coefs=None):
    if loc_str == 'cos':
        return np.cos
    elif loc_str == 'von_mises':
        from scipy.special import iv
        offset = iv(0, kappa) * np.exp(-kappa)
        return lambda x: np.exp(kappa * np.cos(x)) / np.exp(kappa) - offset  # NB: the constant part is in b!
    elif loc_str == 'custom':
        modes = np.arange(1, len(custom_coefs)+1)
        loc_input_ft = custom_coefs / np.sum(custom_coefs)
        return lambda x: 2 * np.sum(loc_input_ft[:,np.newaxis] * np.cos(modes[:,np.newaxis] * x), axis=0)
    else:
        print("Localized input type not implemented")


def d_nonlin_str2fun(nonlin_str):
    if nonlin_str == 'exp':
        return np.exp
    elif nonlin_str == 'relu':
        return lambda x: x > 0
    elif nonlin_str == 'explin':
        return lambda x: np.exp(x) / (1 + np.exp(x))


def load_training_vels():
    training_vels = []
    with open('training_vels.csv', newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            training_vels.append(np.array(row).astype('int').tolist())
    return training_vels


def get_train_set_index_single_vel(vel):
    with open('training_vels.csv') as file:
        reader = csv.reader(file, delimiter=',')
        for train_set, row in enumerate(reader):
            if len(row) == 1 and int(row[0]) == vel:
                return train_set
    return -1


def get_train_set_index(vels):
    with open('training_vels.csv') as file:
        reader = csv.reader(file, delimiter=',')
        for train_set, row in enumerate(reader):
            if len(row) == len(vels):
                row_ints = sorted(int(x) for x in row)
                if row_ints == sorted(vels):
                    return train_set
    return -1


def get_ring_number(s):
    if s == 'double_ring':
        return 2
    elif s == 'three_ring':
        return 3
    else:
        return 1000


def add_noise(s, eps, device):
    import torch
    
    s += eps * torch.randn(s.shape).to(device)
    s[s<0] = 0
    return s


def modes_mask(n_modes, use_modes=None):
    if use_modes is not None:
        mask = np.zeros(2*n_modes-1, dtype='bool')
        for k in use_modes:
            if k == 0:
                mask[0] = True 
            else:
                mask[2*k-1] = True
                mask[2*k] = True
        return mask
    else:
        return np.ones(2*n_modes-1, dtype='bool')


def generate_velocity_trace(time_steps, sigma=0.5, momentum=0.995, dt=1.):
    v = np.zeros(time_steps)
    p = 0.0
    for k in range(time_steps - 1):
        X = (np.random.rand() > p) * np.random.randn() * np.sqrt(dt)
        v[k+1] = sigma * X + momentum * v[k]  # theta_ou = (1 - m) / dt
    return v