import torch
import torch.nn.functional as F
import numpy as np
from operator import itemgetter
from scipy.stats import norm
from scipy.special import iv


class Net(torch.nn.Module):
    def __init__(self, N, J0, b, tau, dt, nonlin, device):
        super(Net, self).__init__()
        self.nonlin = nonlin
        self.N = N
        self.tau = tau
        self.dt = dt
        self.J0 = J0
        self.b = b

    def forward(self, s, inputs):
        inh = self.J0 * s.sum(dim=-1, keepdim=True) / self.N
        f = self.nonlin(inh + inputs + self.b)
        s_new = s + (f - s) * self.dt / self.tau
        return s_new


class FeedbackNet(torch.nn.Module):
    def __init__(self, N, J0, b, tau, dt, nonlin, n_readouts, device):
        super(FeedbackNet, self).__init__()
        self.nonlin = nonlin
        self.N = N
        self.tau = tau
        self.dt = dt
        self.J0 = J0
        self.b = b
        self.out_weights = torch.zeros((n_readouts, N), dtype=torch.float).to(device)
        self.fb_weights = torch.zeros((N, n_readouts), dtype=torch.float).to(device)
        self.in_weights = torch.zeros((N, 1), dtype=torch.float).to(device)

    def rhs(self, s, inputs):
        inh = self.J0 * s.sum(dim=-1, keepdim=True) / self.N
        out = F.linear(s, self.out_weights)
        recurrent_input = F.linear(out, self.fb_weights)
        ext_input = F.linear(inputs, self.in_weights)
        f = self.nonlin(inh + recurrent_input + ext_input + self.b)
        return f - s

    def forward(self, s, inputs):
        inh = self.J0 * s.sum(dim=-1, keepdim=True) / self.N
        out = F.linear(s, self.out_weights)
        recurrent_input = F.linear(out, self.fb_weights)
        ext_input = F.linear(inputs, self.in_weights)
        f = self.nonlin(inh + recurrent_input + ext_input + self.b)
        s_new = s + (f - s) * self.dt / self.tau
        return s_new, out


class LinearNet(torch.nn.Module):
    def __init__(self, N, J0, mu, b, tau, dt, nonlin, n_readouts, device):
        super(LinearNet, self).__init__()
        self.nonlin = nonlin
        self.N = N
        self.tau = tau
        self.dt = dt
        self.J0 = J0
        self.mu = mu
        self.b = torch.from_numpy(np.repeat(b, N)).float().to(device)
        self.out_weights = torch.zeros((n_readouts, N), dtype=torch.float).to(device)
        self.fb_weights = torch.zeros((N, n_readouts), dtype=torch.float).to(device)

    def forward(self, s, inputs):
        dG = self.nonlin(inputs + self.J0 * self.mu + self.b)
        inh = self.J0 * s.sum(dim=1, keepdim=True) / self.N
        out = F.linear(s, self.out_weights)
        feedback = F.linear(out, self.fb_weights)
        f = dG * (inh + feedback)
        s_new = s + (-s + f) * self.dt / self.tau
        return s_new


class LinearClosedLoopNet(torch.nn.Module):
    def __init__(self, N, J0, b, tau, dt, nonlin, n_readouts, device):
        super(LinearClosedLoopNet, self).__init__()
        self.nonlin = nonlin
        self.N = N
        self.tau = tau
        self.dt = dt
        self.J0 = J0
        self.b = torch.from_numpy(np.repeat(b, N)).float().to(device)
        self.in_weights = torch.zeros((N, 1), dtype=torch.float).to(device)
        self.out_weights = torch.zeros((n_readouts, N), dtype=torch.float).to(device)
        self.fb_weights = torch.zeros((N, n_readouts), dtype=torch.float).to(device)

    def forward(self, delta_s, s, v):
        out = F.linear(s.unsqueeze(0), self.out_weights)
        feedback = F.linear(out, self.fb_weights)
        inh = self.J0 * s.sum() / self.N
        dG = self.nonlin(feedback + inh + self.in_weights * v + self.b)

        inh = self.J0 * delta_s.sum(dim=1, keepdim=True) / self.N
        out = F.linear(delta_s, self.out_weights)
        feedback = F.linear(out, self.fb_weights)
        f = dG * (inh + feedback)
        delta_s_new = delta_s + (-delta_s + f) * self.dt / self.tau
        return delta_s_new


def fourier_transform_loc_input(loc_input, kappa=None, custom_coefs=None, n_modes=2):
    loc_input_ft = np.zeros(n_modes)

    if loc_input == 'cos':
        #loc_input_fun = lambda x: cos(x)
        loc_input_ft[1] = 0.5
    elif loc_input == 'von_mises':
        modes = np.arange(n_modes)
        loc_input_ft = iv(modes, kappa) * np.exp(-kappa)
    elif loc_input == 'custom':
        loc_input_ft = np.zeros(n_modes)
        loc_input_ft[1:len(custom_coefs)+1] = custom_coefs
        loc_input_ft = loc_input_ft / np.sum(loc_input_ft)

    offset = loc_input_ft[0]
    loc_input_ft[0] = 0.0

    return loc_input_ft


def generate_weights(params, sim=0):
    N, J0, sigma, r, theta_grid, input_structure, n_modes, loc_input, kappa, custom_coefs = itemgetter('N', 'J0', 'sigma', 'r', 'theta_grid', 'input_structure', 'n_modes', 'loc_input', 'kappa', 'custom_coefs')(params)

    np.random.seed(2874+sim)

    if theta_grid and input_structure == 'double_ring':
        theta = np.linspace(-np.pi, np.pi, N//2, endpoint=False)
        theta = np.hstack((theta,theta))
    elif theta_grid and input_structure == 'three_ring':
        theta = np.linspace(-np.pi, np.pi, N//3, endpoint=False)
        theta = np.hstack((theta,theta,theta))
    elif theta_grid and input_structure == 'random':
        theta = np.linspace(-np.pi, np.pi, N, endpoint=False)
    elif theta_grid and input_structure == 'multi_ring':
        M = int(np.sqrt(N))
        q = np.linspace(0.5/M, 1.0-0.5/M, M)
        w_vals = norm.ppf(q, 0, scale=sigma)
        theta = np.linspace(-np.pi, np.pi, M, endpoint=False)
        shifts = np.arange(M) * 2 * np.pi / M**2
        ww, tt = np.meshgrid(w_vals, theta)
        shifts = np.random.permutation(shifts)
        for i in range(M):
            tt[:,i] += shifts[i]
        w = ww.flatten()
        theta = tt.flatten()
    else:
        theta = 2 * np.pi * np.random.rand(N) - np.pi

    if input_structure == 'double_ring':
        control_weights = sigma * np.ones((N,1))
        control_weights[N//2:] *= -1

    elif input_structure == 'three_ring':
        control_weights = np.zeros((N,1))
        control_weights[:N//3] = sigma
        control_weights[2*N//3:] = -sigma
    elif input_structure == 'multi_ring':
        control_weights = w.reshape(N,1)
    else:
        control_weights = sigma * np.random.randn(N,1)

    loc_input_ft = fourier_transform_loc_input(loc_input, kappa=kappa, n_modes=n_modes, custom_coefs=custom_coefs)

    feedback_weights = np.zeros((2*n_modes-1,N))
    feedback_weights[0,:] = loc_input_ft[0]
    for k in range(1,n_modes):
        feedback_weights[2*k-1,:] = 2 * loc_input_ft[k] * np.cos(k * theta)
        feedback_weights[2*k,:] = 2 * loc_input_ft[k] * np.sin(k * theta)
    feedback_weights *= r

    return feedback_weights, control_weights, theta


def get_nonlin(nonlin_str):
    if nonlin_str == 'relu':
        nonlin = torch.nn.ReLU(inplace=True)
    elif nonlin_str == 'exp':
        nonlin = torch.exp
    elif nonlin_str == 'explin':
        nonlin = lambda x: torch.log1p(torch.exp(x))
    elif nonlin_str == 'tanh':
        nonlin = lambda x: 1.0 + torch.tanh(x)
    elif nonlin_str == 'heaviside':
        nonlin = lambda x: (x > 0).float() + (x > 2).float()
    else:
        raise ValueError(f"Nonlinearity not implemented: {nonlin_str}")
    return nonlin


def init_driven_network(control_weights, feedback_weights, params, device):
    N, J0, b, tau, dt, nonlin_str = itemgetter('N', 'J0', 'b', 'tau', 'dt', 'nonlinearity')(params)
    nonlin = get_nonlin(nonlin_str)
    driven_net = Net(N, J0, b, tau, dt, nonlin, device).to(device)
    in_weights = np.concatenate((feedback_weights, control_weights.reshape(1,N)), axis=0)
    in_weights = torch.from_numpy(in_weights).float().to(device)
    return driven_net, in_weights


def init_feedback_network(control_weights, out_weights, feedback_weights, params, device):
    N, J0, b, tau, dt, nonlin_str, n_modes = itemgetter('N', 'J0', 'b', 'tau', 'dt', 'nonlinearity', 'n_modes')(params)
    nonlin = get_nonlin(nonlin_str)
    feedback_net = FeedbackNet(N, J0, b, tau, dt, nonlin, 2*n_modes-1, device).to(device)
    feedback_net.in_weights = torch.from_numpy(control_weights).float().to(device)
    feedback_net.fb_weights = torch.from_numpy(feedback_weights.T).float().to(device)
    feedback_net.out_weights = torch.from_numpy(out_weights).float().to(device)
    return feedback_net

def init_linear_network(out_weights, feedback_weights, mu, params, device):
    N, J0, b, tau, dt, nonlin_str = itemgetter('N', 'J0', 'b', 'tau', 'dt', 'nonlinearity')(params)
    nonlin = lambda x: torch.exp(x) / (1 + torch.exp(x))
    n_readouts = 2
    linear_net = LinearNet(N, J0, mu, b, tau, dt, nonlin, n_readouts, device).to(device)
    linear_net.fb_weights = torch.from_numpy(feedback_weights.T).float().to(device)
    linear_net.out_weights = torch.from_numpy(out_weights).float().to(device)
    return linear_net

def init_linear_closed_loop_network(control_weights, out_weights, feedback_weights, params, device):
    N, J0, b, tau, dt, nonlin_str = itemgetter('N', 'J0', 'b', 'tau', 'dt', 'nonlinearity')(params)
    nonlin = lambda x: torch.exp(x) / (1 + torch.exp(x))
    n_readouts = 2
    linear_net = LinearClosedLoopNet(N, J0, b, tau, dt, nonlin, n_readouts, device).to(device)
    linear_net.in_weights = torch.from_numpy(control_weights).float().to(device)
    linear_net.fb_weights = torch.from_numpy(feedback_weights.T).float().to(device)
    linear_net.out_weights = torch.from_numpy(out_weights).float().to(device)
    return linear_net