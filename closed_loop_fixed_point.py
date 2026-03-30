import numpy as np
from scipy.optimize import minimize
import config
from theory import ContinuumModel
import os
import utils

def main(out_path):

    n_rings = 50
    model = ContinuumModel(config.sigma, n_rings)
    w, p = model.inputs_weights, model.inputs_probs
    G = utils.nonlin_str2fun(config.nonlinearity)
    dG = utils.d_nonlin_str2fun(config.nonlinearity)

    def mu(s, p):
        return np.sum(p * s)

    def steady_state_loss_fun(x, G, J0, b, w, v):
        s = G(J0 * x + b + w * v)
        return (x - mu(s, p))**2

    vels = config.vels_rad_per_ms
    pop_acts = []
    dG_pop_acts = []
    mean_acts = []

    for v in vels:
        loss = lambda x: steady_state_loss_fun(x, G, config.J0, config.b, w, v)
        res = minimize(loss, 0.25, bounds=[(1e-6,10)], tol=1e-8)
        mu_ss = res.x[0]
        inputs = config.J0 * mu_ss + config.b + w * v
        s = G(inputs)

        mean_acts.append(mu_ss)
        pop_acts.append(s)
        dG_pop_acts.append(dG(inputs))

    mean_acts = np.hstack(mean_acts)
    pop_acts = np.vstack(pop_acts)
    dG_pop_acts = np.vstack(dG_pop_acts)

    np.save(os.path.join(out_path, 'closed_loop', 'w_values.npy'), w)
    np.save(os.path.join(out_path, 'closed_loop', 'w_density.npy'), p)
    np.save(os.path.join(out_path, 'closed_loop', 'mu.npy'), mean_acts)
    np.save(os.path.join(out_path, 'closed_loop', 's.npy'), pop_acts)


    phis = np.load(os.path.join(out_path, 'v_train=360', 'phi.npy'))
    rhos = np.load(os.path.join(out_path, 'v_train=360', 'rho.npy'))

    # phi = phis[:,1]
    # rho = rhos[:,1]

    # prefactor = 2 * config.r * config.custom_coefs[0]
    # readout_weights = rho * np.exp(-1j * np.sign(w) * phi)
    # eigenvalue = prefactor * np.sum(p[np.newaxis,:] * readout_weights[np.newaxis,:] * dG_pop_acts, axis=-1)



if __name__ == "__main__":

    network_params = utils.network_params_str(config.params)
    out_path = os.path.join('theory', network_params)
    if not os.path.exists(os.path.join(out_path, 'closed_loop')):
        os.makedirs(os.path.join(out_path, 'closed_loop'))

    main(out_path)