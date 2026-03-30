import numpy as np

sigma = 50  # 50 for exp + cos,  100 for cos + relu
J0 = -10   #-5 for cos and von mises   # -1 for von mises or exponential nonlin
n_sim = 1
r = 10  # 20 or 10 for von mises, 10 for cos, 3 for exp nonlin
N = 1000
b = 2  #1 or 2 for cos and von mises, -5 for von mises, -2 for exp nonlin
tau = 80
dt = 0.1  # 0.1 or 1.0 (0.1 needed so that readout error accurately captured by continuum model)
theta_grid = True
input_structure = 'random'   # double_ring, three_ring or random or multi_ring
nonlinearity = 'explin'   # exp, relu, explin
loc_input = 'cos'  # custom, von_mises or cos
kappa = 3
custom_coefs = [1.0, 0.5, 0.15]
#skip = 10
data_path = '/mnt/bpd/data'  #'/data/path_integration_networks'
eps = 0.5 #0.25
online = False
lmbda = 0.0
n_modes = 6

params = {'N': N, 'J0': J0, 'b': b, 'tau': tau, 'dt': dt, 'N': N, 'sigma': sigma, 'r': r, 'theta_grid': theta_grid,
          'input_structure': input_structure, 'nonlinearity': nonlinearity, 'n_modes': n_modes, 'loc_input': loc_input,
          'kappa': kappa, 'custom_coefs': custom_coefs, 'eps': eps, 'lmbda': lmbda, 'online': online}

vels = np.arange(1, 31) * 60
n_vels = len(vels)
vels_rad_per_ms = np.hstack((-vels[::-1], vels)) * np.pi / (1000 * 180)

T_sinusoidal = 1000 * np.array([0.25, 0.5, 1.0, 5.0])
x_sinusoidal = np.pi / 4