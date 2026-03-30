import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from warnings import filterwarnings
from scipy.linalg import LinAlgWarning
import argparse
import network
import torch
import utils
import config
import os


def location_input(psi, n_modes):
    time_steps, n_vels = psi.shape
    z_target = torch.ones((time_steps, n_vels, 2*n_modes-1)).float()
    for k in range(1,n_modes):
        z_target[:,:,2*k-1] = torch.cos(k * psi)
        z_target[:,:,2*k] = torch.sin(k * psi)
    return z_target


#filterwarnings(action='error', category=LinAlgWarning, module='sklearn')

# note: training phase starts at input location zero if v = n * v_min
def run_driven_net(driven_net, in_weights, vels, params, device, keep_init=True):
    v_min = torch.min(torch.abs(vels))
    train_steps = int(np.round(2 * np.pi / (params['dt'] * v_min))) + 1
    init_steps = int(np.round(2 * np.pi / (params['dt'] * v_min)))

    turn_ind = np.zeros(len(vels), dtype=int)
    for i, v in enumerate(vels):
        turn_ind[i] = int(np.round(np.abs(2 * np.pi / (params['dt'] * v))))

    time_steps = init_steps + train_steps
    t = params['dt'] * torch.arange(time_steps)
    psi = vels.unsqueeze(0) * t.unsqueeze(1)

    if keep_init:
        s_driven = torch.zeros(time_steps,len(vels),params['N']).to('cpu')
    else:
        s_driven = torch.zeros(train_steps,len(vels),params['N']).to('cpu')
    
    z_target = location_input(psi, params['n_modes'])
    z_target = torch.cat((z_target, vels.unsqueeze(0).repeat(time_steps, 1).unsqueeze(2)),dim=2)

    s = torch.zeros(len(vels), params['N']).to(device)
    for i in range(time_steps):
        if keep_init:
            s_driven[i] = s
        elif i >= init_steps:
            s_driven[i-init_steps] = s

        s = driven_net(s, z_target[i].to(device) @ in_weights)

    s_driven = s_driven.detach().cpu().numpy()
    z_target = z_target.detach().cpu().numpy()
    psi = psi.detach().cpu().numpy()

    return t, s_driven, psi, z_target, turn_ind


def run_driven_net_at_slow_velocity(driven_net, in_weights, vels, params, device, batch_size=10):

    v = torch.abs(vels[1])

    psi0 = np.linspace(-np.pi, np.pi, batch_size, endpoint=False)
    psi0 = torch.from_numpy(psi0).float()
    psi0 = torch.hstack((psi0, psi0))

    vels = vels.clone().detach().repeat_interleave(batch_size)

    init_steps = int(1000 / params['dt'])
    train_steps = int(2 * np.pi / (v * batch_size * params['dt']))
    time_steps = init_steps + train_steps

    t = params['dt'] * torch.arange(time_steps)
    psi = psi0.unsqueeze(0) + vels.unsqueeze(0) * t.unsqueeze(1)
    s_driven = torch.zeros(train_steps,len(vels),params['N']).to('cpu')

    z_target = location_input(psi, params['n_modes'])
    inputs = torch.cat((z_target, vels.unsqueeze(0).repeat(time_steps, 1).unsqueeze(2)),dim=2)

    s = torch.zeros(len(vels), params['N']).to(device)
    for i in range(time_steps):
        if i >= init_steps:
            s_driven[i-init_steps] = s

        s = driven_net(s, inputs[i].to(device) @ in_weights)

    s_driven = s_driven.detach().cpu().numpy()
    z_target = z_target.detach().cpu().numpy()
    psi = psi.detach().cpu().numpy()

    z_target = z_target[-train_steps:]

    return s_driven, psi, z_target


def find_stationary_solutions(driven_net, in_weights, init_steps, params, device, batch_size=360):
    vels = torch.zeros(batch_size).float()
    psi = np.linspace(-np.pi, np.pi, batch_size, endpoint=False)
    psi = torch.from_numpy(psi).float()
    z_target = location_input(psi.reshape(1,-1), params['n_modes']).squeeze()

    inputs = torch.cat((z_target, vels.unsqueeze(1)), dim=1).to(device)

    s = torch.zeros(batch_size, params['N']).to(device)
    for i in range(init_steps):
        s = driven_net(s, inputs @ in_weights)

    psi = psi.cpu().detach().numpy()
    s = s.cpu().detach().numpy()
    z_target = z_target.cpu().detach().numpy()

    return s, psi, z_target


def get_dataset(s_driven, z_driven, vels, train_ind, training_vels, params):
    idx = []
    for v in training_vels:
        idx.append(np.where(vels==v)[0][0])
    n_vels = len(vels)
    idx = [n_vels - j - 1 for j in idx[::-1]] + [j + n_vels for j in idx]

    s_driven = s_driven[:,idx,:]
    z_driven = z_driven[:,idx,:]
    train_ind = train_ind[idx]

    X = []
    z = []
    training_vels = [-v for v in training_vels[::-1]] + training_vels

    for i, (train_ind_vel, train_vel) in enumerate(zip(train_ind, training_vels)):
        delta = int((360 / abs(train_vel)) * ((1000 / params['dt']) / (params['N'] // 2)))
        zi = z_driven[-train_ind_vel::delta,i,:-1]
        si = s_driven[-train_ind_vel::delta,i,:]
        z.append(zi)
        X.append(si)

    X = np.vstack(X)
    z = np.vstack(z)
    return X, z


def get_slow_dataset(s_driven, z_driven, params):

    batch_size = s_driven.shape[1] // 2
    delta = int(s_driven.shape[0] * 2 * batch_size / params['N'])
    z = z_driven[::delta,:,:].copy()
    X = s_driven[::delta,:,:].copy()

    z = z.reshape(z.shape[0] * z.shape[1], z.shape[2])
    X = X.reshape(X.shape[0] * X.shape[1], X.shape[2])

    return X, z


from scipy import linalg
def ridge_svd(X, y, alpha=1e-5):
    alpha = np.asarray(alpha, dtype=X.dtype).ravel()
    U, s, Vt = linalg.svd(X, full_matrices=False)
    s = s[:, np.newaxis]
    UTy = np.dot(U.T, y)
    d = s / (s**2 + alpha)
    d_UT_y = d * UTy
    return np.dot(Vt.T, d_UT_y).T


def fit_readout(X, z, alpha=1e-5):
    coef = ridge_svd(X, z, alpha)
    out = X @ coef.T
    mse = mean_squared_error(z, out)
    return coef, mse


def main(alphas, vary, save_traces, cv):
    device = torch.device('cuda:0')
    control_inputs = torch.from_numpy(config.vels_rad_per_ms).float()
    params = config.params

    if vary == 'sigma':
        sigmas = np.linspace(25,250,10)
        for sigma in sigmas:
            print('sigma = {}'.format(sigma))
            params['sigma'] = sigma
            run_sim(device, control_inputs, params, alphas=alphas, save_traces=save_traces, cv=cv)

    elif vary == 'N':
        Ns = [500, 1000, 2500, 5000]
        for N in Ns:
            print('N = {}'.format(N))
            params['N'] = N
            run_sim(device, control_inputs, params, alphas=alphas, save_traces=save_traces, cv=cv)

    else:
        run_sim(device, control_inputs, params, alphas=alphas, save_traces=save_traces, cv=cv)


def run_sim(device, control_inputs, params, alphas=[100], save_traces=False, cv=False):

    for i in range(config.n_sim):

        print('simulation # {}'.format(i+1))

        path1 = utils.get_data_path(config.data_path, params)
        if not os.path.exists(path1):
            os.makedirs(path1)

        feedback_weights, control_weights, theta = network.generate_weights(params, sim=i)
        driven_net, in_weights = network.init_driven_network(control_weights, feedback_weights, params, device)
        t_driven, s_driven, psi, z_driven, turn_ind = run_driven_net(driven_net, in_weights, control_inputs, params, device)

        path2 = os.path.join(path1, 'sim={}'.format(i))
        if not os.path.exists(path2):
            os.makedirs(path2)

        np.save(os.path.join(path2, 'feedback_weights.npy'), feedback_weights)
        if save_traces:
            np.save(os.path.join(path2, 'z.npy'), z_driven)
            np.save(os.path.join(path2, 'psi.npy'), psi)
            np.save(os.path.join(path2, 't.npy'), t_driven)
            np.save(os.path.join(path2, 's_driven.npy'), s_driven.astype(np.float32))
        np.save(os.path.join(path2, 's_driven_final.npy'), s_driven[-1])
        np.save(os.path.join(path2, 'psi_final.npy'), psi[-1])
        np.save(os.path.join(path2, 'turn_ind.npy'), turn_ind)
        np.save(os.path.join(path2, 'control_weights.npy'), control_weights)
        np.save(os.path.join(path2, 'theta.npy'), theta)
        np.save(os.path.join(path2, 'vels.npy'), np.hstack((-config.vels[::-1], config.vels)))


        init_steps = 1000
        s_stationary, psi_stationary, z_stationary = find_stationary_solutions(driven_net, in_weights, init_steps, params, device)
        np.save(os.path.join(path2, 's_stationary.npy'), s_stationary)
        np.save(os.path.join(path2, 'psi_stationary.npy'), psi_stationary)
        np.save(os.path.join(path2, 'z_stationary.npy'), z_stationary)


        init_steps = 1000
        slow_vels = torch.tensor([-10. * np.pi / (1000 * 180), 10. * np.pi / (1000 * 180)]).float()
        s_slow, psi_slow, z_slow = run_driven_net_at_slow_velocity(driven_net, in_weights, slow_vels, params, device)
        np.save(os.path.join(path2, 's_slow.npy'), s_slow)
        np.save(os.path.join(path2, 'psi_slow.npy'), psi_slow)
        np.save(os.path.join(path2, 'z_slow.npy'), z_slow)


        training_vels_lst = utils.load_training_vels()

        for k, training_vels in enumerate(training_vels_lst):
            print('training velocities: {}'.format(training_vels))

            if len(training_vels) == 1 and training_vels[0] == 10:
                X, z = get_slow_dataset(s_slow, z_slow, params)
            else:
                X, z = get_dataset(s_driven, z_driven, config.vels, turn_ind, training_vels, params)

            # Leave-One-Out cross-validation
            if cv:
                # scoring = neg_mean_squared_error
                reg = RidgeCV(fit_intercept=False, alphas=alphas, gcv_mode='svd', store_cv_values=True)
                reg.fit(X, z)

                cv_path = os.path.join(path2, 'set={}_cv'.format(k))
                if not os.path.exists(cv_path):
                    os.makedirs(cv_path)

                with open(os.path.join(cv_path, 'alpha_opt.txt'), 'w') as f:
                    f.write(str(reg.alpha_))

                with open(os.path.join(cv_path, 'best_score.txt'), 'w') as f:
                    f.write(str(reg.best_score_))

                np.save(os.path.join(cv_path, 'cv_values.npy'), reg.cv_values_)
                np.save(os.path.join(cv_path, 'alpha_values.npy'), alphas)
            

            for alpha in alphas:
                print('alpha = {:.0e}'.format(alpha))

                path3 = os.path.join(path2, 'set={}_alpha={:.0e}'.format(k, alpha))
                if not os.path.exists(path3):
                    os.makedirs(path3)

                try:
                    out_weights, mse = fit_readout(X, z, alpha=alpha)

                except LinAlgWarning:
                    with open(os.path.join(path3, 'fit_successful.txt'), 'w') as f:
                        f.write('0')
                    continue

                with open(os.path.join(path3, 'fit_successful.txt'), 'w') as f:
                    f.write('1')

                np.save(os.path.join(path3, 'readout_weights.npy'), out_weights)
                np.save(os.path.join(path3, 'mse_readout.npy'), mse)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learn path integration networks.')
    parser.add_argument('--alpha', default=[1e-4], type=float, required=False, nargs='+', help='regularization parameter')
    parser.add_argument('--cv', dest='perform_cv', action='store_true', help='cross-validation')
    parser.add_argument('--vary', type=str, required=False, default='no', help='vary sigma, g or N')
    parser.add_argument('--save-traces', dest='save_traces', action='store_true')
    parser.add_argument('--discard-traces', dest='save_traces', action='store_false')
    parser.set_defaults(save_traces=False)
    parser.set_defaults(perform_cv=False)
    args = parser.parse_args()

    main(args.alpha, args.vary, args.save_traces, args.perform_cv)