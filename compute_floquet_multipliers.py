import numpy as np
import scipy
import network
import torch
import utils
import config
import argparse
import os
from simulate_closed_loop import drop_unused_modes
import matplotlib.pyplot as plt
import matplotlib_config


def run_linear_closed_loop_net(linear_net, s, v, params, device):

    time_steps = s.shape[0]

    Rs = []

    delta_s = torch.eye(params['N']).to(device)
    for i in range(time_steps):

        si = torch.from_numpy(s[i]).float().to(device)
        delta_s = linear_net(delta_s, si, v)

        if i % 100 == 0:
            #print(i)
            Q, R = torch.linalg.qr(delta_s.T)
            Rs.append(R.cpu().detach())
            delta_s = Q.T

    Phi = delta_s.T
    for R in Rs[::-1]:
        Phi = Phi @ R.to(device)

    return Phi.detach().cpu().numpy()


def plot_lm_eigenvalues(vels, eigvals, fname='eigenvalues_monodromy_matrix.pdf', fig_path='.'):

    x = np.sort(np.abs(eigvals), axis=-1)[:,::-1]

    fig = plt.figure(figsize=(2.75, 1.5))
    ax = fig.add_axes([0.275, 0.35, 0.4, 0.6]) 

    plt.plot(vels, x[:,0], markersize=3, marker='.', label=r'$|\lambda_1|$')
    plt.plot(vels, x[:,1], markersize=3, marker='.', label=r'$|\lambda_2|$')
    plt.plot(vels, x[:,2], markersize=3, marker='.', label=r'$|\lambda_3|$')

    plt.yscale('log')
    plt.yticks([1e-6, 1e-4, 1e-2, 1]);
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks([120, 360, 600])
    plt.xlabel('Target velocity [deg/s]')
    plt.ylabel(r'$|\lambda|$');

    plt.savefig(os.path.join(fig_path, fname), format='pdf', dpi=300)
    plt.close()


def plot_dominant_eigenvector(theta, x, w_simulation, fname, fig_path):

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    
    fig = plt.figure(figsize=(2.25,1.1))
    ax = fig.add_axes([0.2, 0.35, 0.55, 0.6]) 

    vmax = np.maximum(np.max(w_simulation), -np.min(w_simulation))

    normalize = Normalize(vmin=-vmax, vmax=vmax)
    scalarMap = ScalarMappable(norm=normalize, cmap="coolwarm")

    im=ax.scatter(theta, x, marker='.', edgecolors='none', s=5.0, c=w_simulation, zorder=2, cmap=scalarMap.cmap, vmin=-vmax, vmax=vmax)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels([-180, -90, 0, 90, 180])
    ax.set_xlabel(r'$\theta$ [deg]')
    ax.set_ylabel('Eigenvector component')

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    cax = fig.add_axes([0.775, 0.4, 0.015, 0.475])
    cb=plt.colorbar(im, cax=cax)
    cb.set_label(r'$u$ [a.u]')

    plt.savefig(os.path.join(fig_path, fname), dpi=300, bbox_inches='tight')
    plt.close()


def main(training_set, alpha, use_modes=[1]):

    skip = 10  # network act only saved every 10 steps

    device = torch.device('cuda:0')
    params = config.params

    path1 = utils.get_data_path(config.data_path, params)
    if not os.path.exists(path1):
        print('Data missing for {}'.format(path1))
        return False

    network_params = utils.network_params_str(config.params)

    for i in range(config.n_sim):
        print('Simulation # {}'.format(i+1))

        path2 = os.path.join(path1, 'sim={}'.format(i))
        if not os.path.exists(path2):
            print('Simulation missing!')
            continue

        feedback_weights = np.load(os.path.join(path2, 'feedback_weights.npy'))
        control_weights = np.load(os.path.join(path2, 'control_weights.npy'))
        theta = np.load(os.path.join(path2, 'theta.npy'))

        if not params['online']:
            feedback_weights = drop_unused_modes(feedback_weights, use_modes)

        if params['online']:
            path3 = path2
            fit_successful = True
        else:
            path3 = os.path.join(path2, 'set={}_alpha={:.0e}'.format(training_set, alpha))
            with open(os.path.join(path3, 'fit_successful.txt')) as f:
                fit_successful = int(f.readlines()[0])

        if fit_successful:
            out_weights = np.load(os.path.join(path3, 'readout_weights.npy'))

            s_vels = np.load(os.path.join(path3, 's_vels.npy'), mmap_mode='r')
            psi_vels = np.load(os.path.join(path3, 'psi_vels.npy'))

            if not params['online']:
                out_weights = drop_unused_modes(out_weights, use_modes)

            vels = np.hstack((-config.vels[::-1], config.vels))
            eigvals = []
            dominant_eigenvectors = []

            perturbation_vels = vels[30:40]

            for v_deg_s in perturbation_vels:

                v = v_deg_s / 180 * np.pi / 1000.

                train_id = np.where(vels == v_deg_s)[0][0]

                psi_final = psi_vels[-1,train_id]

                k = psi_final // (2 * np.pi)

                start_index = np.argmin(np.abs(psi_vels[:,train_id] - 2 * np.pi * (k - 1)))
                stop_index = np.argmin(np.abs(psi_vels[:,train_id] - 2 * np.pi * k))

                #psi_initial = psi_final - 2 * np.pi
                #start_index = np.argmin(np.abs(psi_vels[:,train_id] - psi_initial))

                #psi_cycle = psi_vels[start_index:stop_index+1,train_id]
                s_cycle = s_vels[start_index//skip:stop_index//skip+1,train_id,:].copy()
                s_cycle = np.repeat(s_cycle, skip, axis=0)

                linear_net = network.init_linear_closed_loop_network(control_weights.T, out_weights, feedback_weights, params, device)

                monodromy = run_linear_closed_loop_net(linear_net, s_cycle, v, params, device)

                #np.save('monodromy.npy', monodromy)

                S, Z = scipy.linalg.schur(monodromy, output='complex')
                eigvals.append(np.diagonal(S))

                _, dominant_eigenvector = scipy.sparse.linalg.eigs(monodromy, k=1, which='LM')

                if np.mean(dominant_eigenvector[params['N']//2:]) < 0:
                    dominant_eigenvectors.append(dominant_eigenvector.squeeze())
                else:
                    dominant_eigenvectors.append(-dominant_eigenvector.squeeze())

            eigvals = np.stack(eigvals)
            dominant_eigenvectors = np.stack(dominant_eigenvectors)

            # np.save('eigvals.npy', eigvals)
            # np.save('dominant_eigenvectors.npy', dominant_eigenvectors)

            fig_path = os.path.join('figs', network_params, 'stability')
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)

            plot_lm_eigenvalues(perturbation_vels, eigvals, fig_path=fig_path)

            #theo_path = os.path.join('theory', network_params)
            #w = np.load(os.path.join(theo_path, 'w_values.npy'))
            #ds = np.load(os.path.join(theo_path, 'ds.npy'))

            for v_deg_s in perturbation_vels:
                k = np.where(perturbation_vels == v_deg_s)[0][0]
                x = np.real(dominant_eigenvectors[k,:])
                fname = 'monodromy_dominant_eigenvector_v={}.pdf'.format(v_deg_s)
                plot_dominant_eigenvector(theta, x, control_weights, fname, fig_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Compute Floquet multipliers.')
    parser.add_argument('--set', type=int, required=False, default=0, help='set of training velocities')
    parser.add_argument('--alpha', type=float, required=False, default=1e-4, help='regularization parameter')
    args = parser.parse_args()

    main(args.set, args.alpha)