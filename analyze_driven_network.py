import numpy as np
import os
import utils
import argparse
import config


def comp_quant_simulation_doublering(s_driven, psi, turn_ind, theta, n_modes):
    N = len(theta)
    n_vels = len(turn_ind)
    lmbda_plus = np.zeros((n_vels, n_modes))
    lmbda_minus = np.zeros((n_vels, n_modes))
    phase_plus = np.zeros((n_vels, n_modes))
    phase_minus = np.zeros((n_vels, n_modes))
    delta_psi = np.zeros(n_vels)

    for i in range(n_vels):

        skip = turn_ind[i] // 1000
        skip = max(skip, 1)

        psi_vel = psi[-turn_ind[i]::skip,i]
        s_vel = s_driven[-turn_ind[i]-1:-1:skip,i]

        out_plus = np.mean(s_vel[:,:N//2] * np.exp(1j * theta[np.newaxis,:N//2]), axis=1)
        out_minus = np.mean(s_vel[:,N//2:] * np.exp(1j * theta[np.newaxis, N//2:]), axis=1)
        rot_phase = np.angle(out_plus)  #0.5 * (np.angle(out_plus) + np.angle(out_minus))
        dpsi = np.angle(np.exp(1j * (psi_vel - rot_phase)))
        delta_psi[i] = np.angle(np.mean(np.exp(1j*dpsi))) 

        for k in range(n_modes):
            out_plus = np.mean(s_vel[:,:N//2] * np.exp(1j * k * theta[np.newaxis,:N//2]), axis=1)
            out_minus = np.mean(s_vel[:,N//2:] * np.exp(1j * k * theta[np.newaxis, N//2:]), axis=1)
            rot_phase_plus = np.angle(out_plus)
            rot_phase_minus = np.angle(out_minus)
            lmbda_plus[i,k] = np.mean(np.abs(out_plus))
            lmbda_minus[i,k] = np.mean(np.abs(out_minus))
            phase_plus[i,k] = np.mean(np.angle(np.exp(1j * (-k * psi_vel + k *  delta_psi[i] + rot_phase_plus))))
            phase_minus[i,k] = np.mean(np.angle(np.exp(1j * (-k * psi_vel + k *  delta_psi[i] + rot_phase_minus))))

    return lmbda_plus, lmbda_minus, phase_plus, phase_minus, delta_psi

def comp_quant_simulation_random(sim_vels, s_driven, turn_ind, psi_driven, theta, v_train):
    train_id = np.where(sim_vels==v_train)[0][0]
    s = s_driven[-turn_ind[train_id]-1:-1, train_id]
    psi = psi_driven[-turn_ind[train_id]:, train_id]
    psi = np.angle(np.exp(1j * psi))

    out = np.mean(s * np.exp(1j * theta[np.newaxis,:]), axis=1).squeeze()
    bump_phase = np.angle(out)
    delta_psi = np.angle(np.exp(1j*(psi - bump_phase)))
    lmbda = np.zeros((len(theta), config.n_modes))
    xi = np.zeros((len(theta), config.n_modes))
    for k in range(config.n_modes):
        out = np.mean(s * np.exp(1j * k * psi[:,np.newaxis] ), axis=0).squeeze()
        lmbda[:,k] = np.abs(out)
        xi[:,k] = np.angle(np.exp(1j * (-np.angle(out) + k * theta[np.newaxis,:] - k * np.mean(delta_psi))))
    return lmbda, xi, delta_psi

def readout_error(s, psi, w):
    out = w @ s.T
    z = out[0] + 1j * out[1]
    z *= np.exp(-1j * psi)
    z = np.mean(z, axis=-1)
    return np.real(z) - 1, np.imag(z)

def mode0_readout_error(s, w):
    out = w @ s.T
    x = np.mean(out, axis=-1)
    return np.real(x) - 1 

def comp_readout_error_simulation(s_driven, psi, turn_ind, readout_weights, test_vels, n_modes):
    n_test = len(test_vels)

    radial_errors = np.zeros((n_modes, n_test))
    angular_errors = np.nan * np.zeros((n_modes, n_test))

    for j in range(n_test):

        skip = turn_ind[j] // 100
        s = s_driven[-turn_ind[j]::skip,j]
        psi_vel = psi[-turn_ind[j]::skip,j]

        radial_errors[0,j] = mode0_readout_error(s, readout_weights[0])
        for m in range(1, n_modes):
            w = readout_weights[[2 * m - 1, 2 * m]]
            radial_error, angular_error  = readout_error(s, m * psi_vel, w)
            radial_errors[m,j], angular_errors[m,j] = radial_error, angular_error

    return radial_errors, angular_errors


def save_fourier(out_path, sim_vels, lmbda_plus, lmbda_minus, xi_plus, xi_minus, delta_psi):
    np.save(os.path.join(out_path, 'vels.npy'), sim_vels)
    np.save(os.path.join(out_path, 'delta_psi.npy'), delta_psi)
    np.save(os.path.join(out_path, 'lmbda_minus.npy'), lmbda_minus)
    np.save(os.path.join(out_path, 'lmbda_plus.npy'), lmbda_plus)
    np.save(os.path.join(out_path, 'xi_minus.npy'), xi_minus)
    np.save(os.path.join(out_path, 'xi_plus.npy'), xi_plus)

def save_fourier_random(out_path, lmbda, xi, delta_psi):
    np.save(os.path.join(out_path, 'delta_psi.npy'), delta_psi)
    np.save(os.path.join(out_path, 'lmbda.npy'), lmbda)
    np.save(os.path.join(out_path, 'xi.npy'), xi)

def save_readout_errors(out_path, radial_errors, tangential_errors ):
    np.save(os.path.join(out_path, 'radial_errors.npy'), radial_errors)
    np.save(os.path.join(out_path, 'tangential_errors.npy'), tangential_errors)


def analyse_doublering(sim_vels, s_driven, turn_ind, psi, theta, train_vels, readout_weights, out_path):

    lmbda_plus, lmbda_minus, xi_plus, xi_minus, delta_psi = comp_quant_simulation_doublering(s_driven, psi, turn_ind, theta, config.n_modes)
    save_fourier(out_path, sim_vels, lmbda_plus, lmbda_minus, xi_plus, xi_minus, delta_psi)

    for k, v in enumerate(train_vels):
        readout_path = os.path.join(out_path, 'v_train={}'.format(v))
        if not os.path.exists(readout_path):
            os.makedirs(readout_path)

        radial_errors, tangential_errors = comp_readout_error_simulation(s_driven, psi, turn_ind, readout_weights[k], sim_vels, config.n_modes)
        save_readout_errors(readout_path, radial_errors, tangential_errors)


def analyse_continuum(sim_vels, s_driven, turn_ind, psi, theta, control_weights, train_vels, readout_weights, out_path):

    for k, v in enumerate(train_vels):

        out_path = os.path.join(out_path, 'v_train={}'.format(v))
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        lmbda, xi, delta_psi = comp_quant_simulation_random(sim_vels, s_driven, turn_ind, psi, theta, v)
        save_fourier_random(out_path, lmbda, xi, delta_psi)

        radial_errors, tangential_errors = comp_readout_error_simulation(s_driven, psi, turn_ind, readout_weights[k], sim_vels, config.n_modes)
        save_readout_errors(out_path, radial_errors, tangential_errors)


def main():
    parser = argparse.ArgumentParser(description='Compute macroscopic quantities.')
    parser.add_argument('--sim', type=int, required=False, default=0, help='simulation number')
    parser.add_argument('--alpha', type=float, required=False, default=1e-4, help='regularization parameter')
    parser.add_argument('--vels', type=int, required=False, nargs='+', default=[360], help='training velocity')
    args = parser.parse_args()

    network_params = utils.network_params_str(config.params)
    out_path = os.path.join('simulation', network_params, 'sim={}'.format(args.sim))
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    data_path = utils.get_data_path(config.data_path, config.params)
    data_path = os.path.join(data_path, 'sim={}'.format(args.sim))

    try:

        s_driven = np.load(os.path.join(data_path, 's_driven.npy'), mmap_mode='r')
        turn_ind = np.load(os.path.join(data_path, 'turn_ind.npy'))
        psi = np.load(os.path.join(data_path, 'psi.npy'))

        theta = np.load(os.path.join(data_path, 'theta.npy'))
        control_weights = np.load(os.path.join(data_path, 'control_weights.npy')).squeeze()

        readout_weights = np.zeros((len(args.vels), 2 * config.n_modes - 1, config.N))
        fb_weights = np.zeros((len(args.vels), 2 * config.n_modes - 1, config.N))

        for k in range(len(args.vels)):

            train_set = utils.get_train_set_index_single_vel(args.vels[k])

            fit_path = os.path.join(data_path, 'set={}_alpha={:.0e}'.format(train_set, args.alpha))

            with open(os.path.join(fit_path, 'fit_successful.txt')) as f:
                fit_successful = int(f.readlines()[0])

            if fit_successful:
                readout_weights[k] = np.load(os.path.join(fit_path, 'readout_weights.npy'))
                fb_weights[k] = np.load(os.path.join(data_path, 'feedback_weights.npy'))

    except FileNotFoundError:
        print('Simulation data does not exist.')
        exit()

    sim_vels = np.hstack((-config.vels[::-1], config.vels))

    if config.input_structure == 'double_ring':
        analyse_doublering(sim_vels, s_driven, turn_ind, psi, theta, args.vels, readout_weights, out_path)
    else:
        analyse_continuum(sim_vels, s_driven, turn_ind, psi, theta, control_weights, args.vels, readout_weights, out_path)


if __name__ == "__main__":
    main()