import numpy as np
from sklearn.metrics import r2_score
import network
import torch
import utils
import config
import argparse
import os


def comp_psi(out, use_modes=None, online=False):
    if online:
        return np.angle(out[:,:,0] + 1j * out[:,:,1])
    else:
        if use_modes is None:
            return np.angle(out[:,:,1] + 1j * out[:,:,2])
        elif 0 not in use_modes and 1 in use_modes:
            return np.angle(out[:,:,0] + 1j * out[:,:,1])
        else:
            raise ValueError('First Fourier mode not included in use_modes: {}'.format(use_modes))


def comp_readout_dim(n_modes, use_modes, online):
    if online:
        return 2 * len(use_modes)
    else:
        if use_modes is None:
            return 2 * n_modes - 1
        elif 0 not in use_modes:
            return 2 * len(use_modes)
        else:
            return 2 * len(use_modes) - 1


def run_feedback_net(feedback_net, s, inputs, time_steps, params, device, skip=10, use_modes=None, online=False):

    if inputs.dim() == 1:
        M = len(inputs)
        inputs = inputs.reshape(M,1).repeat(1,time_steps).to(device)
    else:
        M = inputs.shape[0]
        inputs = inputs.to(device)

    n_save = time_steps // skip + 1
    s_sim = torch.zeros((n_save, M, params['N'])) #.to(device)
    s_sim[0] = s
    readout_dim = comp_readout_dim(params['n_modes'], use_modes, online)
    out = torch.zeros((time_steps + 1, M, readout_dim)).to(device)

    s = s.to(device)
    for i in range(time_steps):
        s, out_new = feedback_net(s, inputs[:,i].reshape(M,1))

        out[i+1] = out_new

        if (i-1) % skip == 0:
            s_sim[i // skip + 1] = s.cpu().detach()

    s_sim = s_sim.numpy()
    out = out.cpu().detach().numpy()
    psi_sim = comp_psi(out, use_modes=use_modes, online=online)
    psi_sim[0] = psi_sim[1]
    psi_sim = np.unwrap(psi_sim, axis=0)
    t = feedback_net.dt * np.arange(time_steps + 1)

    return t, s_sim, psi_sim


def eval_integration_performance(t, psi_simulation, vels, turn_ind):
    M = len(turn_ind)
    avg_vels = np.zeros(M)
    r2_scores = np.zeros(M)
    psi_theo = vels[np.newaxis,:] * t[:,np.newaxis]
    for i in range(M):
        #T = t[-1] - t[-turn_ind[i]]
        #delta_psi = psi_simulation[-1,i] - psi_simulation[-turn_ind[i],i]

        v = vels[i] / 180 * np.pi / 1000.
        psi_final = psi_simulation[-1,i]
        k = np.abs(psi_final) // (2 * np.pi)
        start_index = np.argmin(np.abs(psi_simulation[:,i] - 2 * np.pi * (k - 1) * np.sign(v)))
        stop_index = np.argmin(np.abs(psi_simulation[:,i] - 2 * np.pi * k * np.sign(v)))

        x = psi_simulation[start_index:stop_index+1,i]
        y = psi_theo[start_index:stop_index+1,i]
        T = t[stop_index] - t[start_index]

        delta_psi = x[-1] - x[0]
        avg_vels[i] = delta_psi * (180 / np.pi) * (1000. / T)

        #x = psi_simulation[-turn_ind[i]:,i]
        #y = psi_theo[-turn_ind[i]:,i]
        p1 = np.poly1d(np.polyfit(x, y, 1, rcond=1e-9))
        r2_scores[i] = r2_score(y, p1(x))

    return r2_scores, avg_vels


def location_input(psi, n_modes):
    batch_size = psi.shape[0]
    z_target = torch.ones((batch_size, 2*n_modes-1)).float()
    for k in range(1,n_modes):
        z_target[:,2*k-1] = torch.cos(k * psi)
        z_target[:,2*k] = torch.sin(k * psi)
    return z_target


def eval_memory_performance(driven_net, feedback_net, in_weights, init_steps, eval_steps, params, device, batch_size=120, use_modes=None):
    vels = torch.zeros(batch_size).float()
    psi = np.linspace(-np.pi, np.pi, batch_size, endpoint=False)
    psi = torch.from_numpy(psi).float()
    #z_target = torch.stack((torch.cos(psi), torch.sin(psi), vels), dim=1).to(device)
    z_target = location_input(psi, params['n_modes'])
    if params['online']:
        mask = utils.modes_mask(params['n_modes'], use_modes=use_modes)
        z_target = z_target[:,mask]
    z_target = torch.cat((z_target, vels.unsqueeze(1)), dim=1).to(device)

    s = torch.zeros(batch_size, params['N']).to(device)
    for i in range(init_steps):
        s = driven_net(s, z_target @ in_weights)

    t_cl, s_cl, psi_cl = run_feedback_net(feedback_net, s, vels.to(device), eval_steps, params, device, skip=100, use_modes=use_modes)

    delta_psi = psi_cl[-1, :] - psi_cl[0, :]
    delta_psi = np.angle(np.exp(1j * delta_psi))
    mse = np.mean(delta_psi**2)

    psi = psi.cpu().detach().numpy()
    return psi, t_cl, s_cl, psi_cl, mse


def main(training_set, alpha, vary, use_modes, save_act, use_mean_readout, use_analytic_readout):
    device = torch.device('cuda:0')
    params = config.params

    if vary == 'sigma':
        sigmas = np.linspace(25,250,10)
        for sigma in sigmas:
            print('sigma = {}'.format(sigma))
            params['sigma'] = sigma
            run_sim(training_set, alpha, device, params, use_modes=use_modes, save_act=save_act, use_mean_readout=use_mean_readout, use_analytic_readout=use_analytic_readout)

    elif vary == 'g':
        gs = np.array([0, 0.5])  #np.linspace(0,1,11)
        for g in gs:
            print('g = {:.2f}'.format(g))
            params['g'] = g
            run_sim(training_set, alpha, device, params, use_modes=use_modes, save_act=save_act, use_mean_readout=use_mean_readout, use_analytic_readout=use_analytic_readout)

    elif vary == 'N':
        Ns = [500, 1000, 2500, 5000]
        for N in Ns:
            print('N = {}'.format(N))
            params['N'] = N
            run_sim(training_set, alpha, device, params, use_modes=use_modes, save_act=save_act, use_mean_readout=use_mean_readout, use_analytic_readout=use_analytic_readout)

    else:
        run_sim(training_set, alpha, device, params, use_modes=use_modes, save_act=save_act, use_mean_readout=use_mean_readout, use_analytic_readout=use_analytic_readout)


def comp_mean_readout(theta, control_weights, p, w, rho, phi):

    mask = control_weights.squeeze() > 0

    mean_rho = np.sum(p * rho)
    mean_phi = np.sum(p * np.abs(phi))

    N = len(theta)

    mean_out_weights = mean_rho * np.ones(N, dtype='complex')
    mean_out_weights *= np.exp(1j * theta) / (N//2)
    mean_out_weights[mask] *= np.exp(1j * mean_phi)
    mean_out_weights[np.logical_not(mask)] *= np.exp(-1j * mean_phi)

    mean_out_weights = np.vstack((np.real(mean_out_weights), np.imag(mean_out_weights)))

    return mean_out_weights


def comp_analytic_readout(theta, control_weights, w, rho, phi):
    N = len(theta)
    weights_theo = np.zeros((2, N))
    for k, w_sample in enumerate(control_weights):
        w_ind = np.argmin(np.abs(w_sample-w))
        z = rho[w_ind] * np.exp(1j * (theta[k] + phi[w_ind])) / (N//2)
        weights_theo[0,k] =  np.real(z)
        weights_theo[1,k] =  np.imag(z)
    return weights_theo


def drop_unused_modes(weights, use_modes):
    readout_dim = weights.shape[0]
    if use_modes is not None:
        mask = np.zeros(readout_dim, dtype='bool')
        for k in use_modes:
            if k == 0:
                mask[0] = True 
            else:
                mask[2*k-1] = True
                mask[2*k] = True
        return weights[mask, :]
    else:
        return weights


def run_sim(training_set, alpha, device, params, use_modes=None, save_act=False, use_mean_readout=False, use_analytic_readout=False, init=True):

    path1 = utils.get_data_path(config.data_path, params)
    if not os.path.exists(path1):
        print('Data missing for {}'.format(path1))
        return False

    vels = np.hstack((-config.vels[::-1], config.vels))

    if params['online']:
        np.random.seed(4385)
    elif params['input_structure'] == 'double_ring':
        np.random.seed(598)
    else:
        np.random.seed(3731)

    for i in range(config.n_sim):
        print('Simulation # {}'.format(i+1))

        path2 = os.path.join(path1, 'sim={}'.format(i))
        if not os.path.exists(path2):
            print('Simulation missing!')
            continue

        if params['online']:
            s_driven = np.random.randn(2 * config.n_vels, params['N'])
        else:
            s_driven = np.load(os.path.join(path2, 's_driven_final.npy'))

        turn_ind = np.load(os.path.join(path2, 'turn_ind.npy'))
        feedback_weights = np.load(os.path.join(path2, 'feedback_weights.npy'))
        control_weights = np.load(os.path.join(path2, 'control_weights.npy'))
        theta = np.load(os.path.join(path2, 'theta.npy'))


        # TMP
        # G = utils.nonlin_str2fun(config.nonlinearity)
        # network_params = utils.network_params_str(config.params)
        # mu = np.load(os.path.join('theory', network_params, 'closed_loop', 'mu.npy'))
        # s_ss = np.load(os.path.join('theory', network_params, 'closed_loop', 's.npy'))
        # s_driven = G(config.J0 * mu[:,np.newaxis] + config.b + control_weights.T * config.vels_rad_per_ms[:,np.newaxis])


        if not params['online']:
            feedback_weights_cl = drop_unused_modes(feedback_weights, use_modes)
        else:
            feedback_weights_cl = feedback_weights

        if params['online']:
            path3 = path2
            fit_successful = True
        else:
            path3 = os.path.join(path2, 'set={}_alpha={:.0e}'.format(training_set, alpha))
            with open(os.path.join(path3, 'fit_successful.txt')) as f:
                fit_successful = int(f.readlines()[0])

        if fit_successful:
            out_weights = np.load(os.path.join(path3, 'readout_weights.npy'))

            if not params['online']:
                out_weights = drop_unused_modes(out_weights, use_modes)

            if use_mean_readout:
                network_params = utils.network_params_str(config.params)
                theory_path = os.path.join('theory', network_params)
                w = np.load(os.path.join(theory_path, 'w_values.npy'))
                p = np.load(os.path.join(theory_path, 'w_density.npy'))
                rho = np.load(os.path.join(theory_path, 'rho.npy'))
                phi = np.load(os.path.join(theory_path, 'phi.npy'))

                out_weights = comp_mean_readout(theta, control_weights, p, w, rho, phi)

            if use_analytic_readout:
                network_params = utils.network_params_str(config.params)
                theory_path = os.path.join('theory', network_params)
                w = np.load(os.path.join(theory_path, 'w_values.npy'))
                rho = np.load(os.path.join(theory_path, 'rho.npy'))
                phi = np.load(os.path.join(theory_path, 'phi.npy'))

                out_weights = comp_analytic_readout(theta, control_weights, w, rho, phi)

            control_weights = control_weights.reshape(params['N'],1)

            print('Test linear integration performance')
            feedback_net = network.init_feedback_network(control_weights, out_weights, feedback_weights_cl, params, device)

            # simulate closed loop system for the period needed for one full turn at the slowest speed + initialization time = 10s
            s0 = torch.from_numpy(s_driven).float().to(device)

            if init:
                init_steps = int(10000 / params['dt']) 
                time_steps = max(turn_ind) + init_steps
                #s0 = utils.add_noise(s0, config.eps, device)
            else:
                time_steps = max(turn_ind)

            skip = 10
            control_inputs = torch.from_numpy(config.vels_rad_per_ms).float()
            t, s, psi = run_feedback_net(feedback_net, s0, control_inputs.to(device), time_steps, params, device, skip=skip, use_modes=use_modes, online=params['online'])
            
            #if not params['online']:
            r2_scores, avg_vels = eval_integration_performance(t, psi, vels, turn_ind)

            if save_act:
                np.save(os.path.join(path3, 's_vels.npy'), s)
            np.save(os.path.join(path3, 'psi_vels.npy'), psi)
            np.save(os.path.join(path3, 't_vels.npy'), t)
            #if not params['online']:
            np.save(os.path.join(path3, 'r2_scores.npy'), r2_scores)
            np.save(os.path.join(path3, 'avg_vels.npy'), avg_vels)


            #perturbations
            time_steps = int(round(1000 / config.dt))
            path4 = os.path.join(path3, 'perturbations')
            os.makedirs(path4, exist_ok=True)
            for k in range(10):
                s0 = torch.from_numpy(s_driven).float().to(device)
                s0 = utils.add_noise(s0, config.eps, device)
                t, s, psi = run_feedback_net(feedback_net, s0, control_inputs.to(device), time_steps, params, device, skip=1, use_modes=use_modes, online=params['online'])
                np.save(os.path.join(path4, 's_{}.npy'.format(k)), s)
                np.save(os.path.join(path4, 'psi_{}.npy'.format(k)), psi)


            feedback_net.dt = 1.0  # for computational reasons
            time_steps = int(30000 / feedback_net.dt)
            v = utils.generate_velocity_trace(time_steps, sigma=0.1, momentum=0.999) / 1000
            control_inputs = torch.from_numpy(v).float().reshape(1,-1)

            s0 = torch.randn(1, params['N']).float().to(device)

            t, s, psi = run_feedback_net(feedback_net, s0, control_inputs.to(device), time_steps, params, device, skip=skip, use_modes=use_modes, online=params['online'])

            np.save(os.path.join(path3, 'velocity_trace.npy'), v)
            np.save(os.path.join(path3, 's_velocity_trace.npy'), s)
            np.save(os.path.join(path3, 'psi_velocity_trace.npy'), psi)
            np.save(os.path.join(path3, 't_velocity_trace.npy'), t)


            init_steps = int(1000 / params['dt']) 
            driven_net, in_weights = network.init_driven_network(control_weights, feedback_weights, params, device)

            print('Test memory performance')
            feedback_net.dt = 1.0  # for computational reasons
            eval_steps = int(300000 / feedback_net.dt)
            psi_init, t_memory, s_memory, psi_memory, mse_memory = eval_memory_performance(driven_net, feedback_net, in_weights, init_steps, eval_steps, params, device, use_modes=use_modes)
            feedback_net.dt = config.dt

            if save_act:
                np.save(os.path.join(path3, 's_memory.npy'), s_memory)
            np.save(os.path.join(path3, 't_memory.npy'), t_memory)
            np.save(os.path.join(path3, 'psi_memory.npy'), psi_memory)
            np.save(os.path.join(path3, 'psi_init.npy'), psi_init)
            np.save(os.path.join(path3, 'mse_memory.npy'), mse_memory)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Simulate closed loop system.')
    parser.add_argument('--set', type=int, required=False, default=0, help='set of training velocities')
    parser.add_argument('--alpha', type=float, required=False, default=1e-4, help='regularization parameter')
    parser.add_argument('--vary', type=str, required=False, default='no', help='vary N, sigma or g')
    parser.add_argument('--save-act', dest='save_act', action='store_true')
    parser.add_argument('--use-mean-readout', dest='use_mean_readout', action='store_true')
    parser.add_argument('--use-analytic-readout', dest='use_analytic_readout', action='store_true')
    parser.add_argument('--discard-act', dest='save_act', action='store_false')
    parser.add_argument('--modes', type=int, nargs='+', default=None, required=False)
    parser.set_defaults(save_act=False)
    parser.set_defaults(use_mean_readout=False)
    parser.set_defaults(use_analytic_readout=False)
    args = parser.parse_args()

    main(args.set, args.alpha, args.vary, args.modes, args.save_act, args.use_mean_readout, args.use_analytic_readout)