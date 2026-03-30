import numpy as np
import network
import os
import utils
import argparse
import config
import theory


def save_fourier(out_path, theo_vels, threshold, lmbda_minus, lmbda_plus, xi_minus, xi_plus, delta_psi):
    np.save(os.path.join(out_path, 'vels.npy'), theo_vels)
    np.save(os.path.join(out_path, 'threshold.npy'), threshold)
    np.save(os.path.join(out_path, 'delta_psi.npy'), delta_psi)
    np.save(os.path.join(out_path, 'lmbda_minus.npy'), lmbda_minus)
    np.save(os.path.join(out_path, 'lmbda_plus.npy'), lmbda_plus)
    np.save(os.path.join(out_path, 'xi_minus.npy'), xi_minus)
    np.save(os.path.join(out_path, 'xi_plus.npy'), xi_plus)

def save_rho_phi(out_path, rho, phi):
    np.save(os.path.join(out_path, 'rho.npy'), rho)
    np.save(os.path.join(out_path, 'phi.npy'), phi)

def save_readout_errors(out_path, radial_errors, tangential_errors):
    np.save(os.path.join(out_path, 'radial_errors.npy'), radial_errors)
    np.save(os.path.join(out_path, 'tangential_errors.npy'), tangential_errors)

def get_rho_phi_doublering(theo, v, alpha):
    rhos = np.zeros((1, config.n_modes))
    phis = np.zeros((1, config.n_modes))
    for m in range(config.n_modes):
        alpha_adj = alpha / (config.N//2)**2
        rho, phi = theo.comp_readout_params(v, m, alpha_adj)
        rhos[0,m], phis[0,m] = rho, phi
    return rhos, phis

def get_rho_phi_continuum(theo, v, alpha):
    _, w = theo.get_input_params()
    rhos = np.zeros((len(w), config.n_modes))
    phis = np.zeros((len(w), config.n_modes))
    for m in range(config.n_modes):
        alpha_adj = alpha / (config.N//2)**2
        rho, phi = theo.comp_readout_params(v, m, alpha_adj)
        rhos[:,m] = rho
        phis[:,m] = phi
    return rhos, phis

def get_connectivity_doublering(rhos, phis):
    theta = network.generate_weights(config.params)[-1]

    loc_input_ft = network.fourier_transform_loc_input(config.loc_input, kappa=config.kappa, n_modes=config.n_modes, custom_coefs=config.custom_coefs)

    k = np.arange(config.n_modes)
    hk = loc_input_ft.copy()
    hk[1:] *= 2

    connectivity_minus = np.sum(hk[:,np.newaxis] * rhos.T * np.cos(k[:,np.newaxis] * theta[np.newaxis,:] - phis.T ), axis=0)
    connectivity_minus *= 2 * config.r / config.N

    connectivity_plus = np.sum(hk[:,np.newaxis] * rhos.T * np.cos(k[:,np.newaxis] * theta[np.newaxis,:] + phis.T ), axis=0)
    connectivity_plus *= 2 * config.r / config.N

    return connectivity_minus, connectivity_plus

def get_readout_error(theo, v_train, alpha):
    n_test_theo = len(theo.vels)
    radial_errors = np.zeros((config.n_modes, n_test_theo))
    tangential_errors = np.nan * np.zeros((config.n_modes, n_test_theo))

    alpha_adj = alpha / (config.N//2)**2
    for k in range(config.n_modes):
        _, radial_error, tangential_error = theo.comp_readout_error(v_train, k, alpha_adj)
        radial_errors[k] = radial_error
        if k > 0:
            tangential_errors[k] = tangential_error

    return radial_errors, tangential_errors

def get_connectivity_continuum(rhos, phis):
    theta = network.generate_weights(config.params)[-1]

    loc_input_ft = network.fourier_transform_loc_input(config.loc_input, kappa=config.kappa, n_modes=config.n_modes, custom_coefs=config.custom_coefs)

    k = np.arange(config.n_modes)
    hk = loc_input_ft.copy()
    hk[1:] *= 2

    k = np.arange(config.n_modes)
    connectivity = np.sum(hk[np.newaxis,:,np.newaxis] * rhos[:,:,np.newaxis] * np.cos(k[np.newaxis,:,np.newaxis] * theta[np.newaxis,np.newaxis,:] - phis[:,:,np.newaxis] ), axis=1)
    connectivity *= 2 * config.r / config.N

    return connectivity


def solve_doublering(v_train, alpha, out_path):

    n_rings = 2
    if config.nonlinearity == 'relu' and config.input_structure == 'double_ring' and config.loc_input == 'cos':
        theo = theory.DoubleRingReLUCosTheory(config.J0, config.r, config.b, config.sigma, config.tau, config.n_modes, n_rings)
    else:
        G = utils.nonlin_str2fun(config.nonlinearity)
        dG = utils.d_nonlin_str2fun(config.nonlinearity)
        h = utils.loc_input_str2fun(config.loc_input, config.kappa, config.custom_coefs)
        theo = theory.DoubleRingGeneralTheory(config.J0, config.r, config.b, config.sigma, config.tau, config.n_modes, n_rings, G, dG, h)

    v_max = 0.05  #0.055
    vels, h, lmbda_minus, lmbda_plus, xi_minus, xi_plus, delta_psi = theo.fourier_transform(v_max)

    save_fourier(out_path, vels, h, lmbda_minus, lmbda_plus, xi_minus, xi_plus, delta_psi)

    for v in v_train:
        readout_path = os.path.join(out_path, 'v_train={}'.format(v))
        os.makedirs(readout_path, exist_ok=True)

        rhos, phis = get_rho_phi_doublering(theo, v, alpha)
        save_rho_phi(readout_path, rhos, phis)

        radial_errors, tangential_errors = get_readout_error(theo, v, alpha)
        save_readout_errors(readout_path, radial_errors, tangential_errors)

        connectivity_minus, connectivity_plus = get_connectivity_doublering(rhos, phis)
        np.save(os.path.join(readout_path, 'connectivity_minus.npy'), connectivity_minus)
        np.save(os.path.join(readout_path, 'connectivity_plus.npy'), connectivity_plus)


def solve_continuum(v_train, alpha, out_path):

    n_rings = 50
    if config.nonlinearity == 'relu' and (config.input_structure == 'random' or config.input_structure == 'multi_ring') and config.loc_input == 'cos':
        theo = theory.ContinuumReLUCosTheory(config.J0, config.r, config.b, config.sigma, config.tau, config.n_modes, n_rings)
    else:
        G = utils.nonlin_str2fun(config.nonlinearity)
        dG = utils.d_nonlin_str2fun(config.nonlinearity)
        h = utils.loc_input_str2fun(config.loc_input, config.kappa, config.custom_coefs)
        theo = theory.ContinuumGeneralTheory(config.J0, config.r, config.b, config.sigma, config.tau, config.n_modes, n_rings, G, dG, h)

    v_max = 0.05
    vels, h, lmbda_minus, lmbda_plus, xi_minus, xi_plus, delta_psi = theo.fourier_transform(v_max)

    w_density, w = theo.get_input_params()

    save_fourier(out_path, vels, h, lmbda_minus, lmbda_plus, xi_minus, xi_plus, delta_psi)
    np.save(os.path.join(out_path, 'w_values.npy'), w)
    np.save(os.path.join(out_path, 'w_density.npy'), w_density)

    for v in v_train:
        readout_path = os.path.join(out_path, 'v_train={}'.format(v))
        os.makedirs(readout_path, exist_ok=True)

        rho, phi = get_rho_phi_continuum(theo, v, alpha)
        save_rho_phi(readout_path, rho, phi)

        radial_errors, tangential_errors = get_readout_error(theo, v, alpha)
        save_readout_errors(readout_path, radial_errors, tangential_errors)

        connectivity = get_connectivity_continuum(rho, phi)
        np.save(os.path.join(readout_path, 'connectivity.npy'), connectivity)


def main():

    network_params = utils.network_params_str(config.params)

    out_path = os.path.join('theory', network_params)
    os.makedirs(out_path, exist_ok=True)

    parser = argparse.ArgumentParser(description='Compute macroscopic quantities.')
    parser.add_argument('--alpha', type=float, required=False, default=1e-4, help='regularization parameter')
    parser.add_argument('--vels', type=int, required=False, nargs='+', default=[360], help='training velocity')
    args = parser.parse_args()


    if config.input_structure == 'double_ring':
        solve_doublering(args.vels, args.alpha, out_path)
    else:
        solve_continuum(args.vels, args.alpha, out_path)


if __name__ == "__main__":
    main()