import numpy as np
import config
import os
import utils
import argparse
import matplotlib.pyplot as plt
import matplotlib_config
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.optimize import minimize
import matplotlib.ticker as mticker


def comp_max_eigval(x, W_rec, control_weights, v_in, dG):

    inputs = x + config.b + control_weights * v_in
    D = np.diag(dG(inputs))
    J = (-np.eye(config.N) + D @ W_rec) / config.tau
    eigenvalues, eigenvectors = np.linalg.eig(J)
    ordering = np.argsort(np.real(eigenvalues))[::-1]
    eigenvalues = eigenvalues[ordering]
    eigenvectors = eigenvectors[:,ordering]

    return np.real(eigenvalues[0])


def loss(s, v_in, feedback_weights, readout_weights, control_weights, G):
    recurrent_input = config.J0 * np.mean(s) + config.b
    out = readout_weights @ s
    feedback_input = feedback_weights.T @ out
    ext_input = control_weights * v_in
    f = G(recurrent_input + feedback_input + ext_input)
    F = np.mean((f - s)**2)
    return F


# def d_loss(s, v_in, feedback_weights, readout_weights, control_weights, G, dG):
#     W_rec = feedback_weights.T @ readout_weights + config.J0 / config.N
#     x = W_rec @ s
#     inputs = x + config.b + control_weights * v_in
#     D = np.diag(dG(inputs))
#     J = (-np.eye(config.N) + D @ W_rec) / config.tau
#     return J


def plot_pattern(theta, control_weights, act_pattern, v, fig_path, network_params):

    theo_path = os.path.join('theory', network_params, 'closed_loop')
    w = np.load(os.path.join(theo_path, 'w_values.npy'))
    s_bar = np.load(os.path.join(theo_path, 's.npy'))

    fname = 'homogeneous_state_v={}.pdf'.format(v)

    fig = plt.figure(figsize=(2.25,1.1))
    ax = fig.add_axes([0.2, 0.35, 0.55, 0.6]) 

    #vmax = np.maximum(np.max(control_weights), -np.min(control_weights))
    vmax= 3 * config.sigma
    im = ax.scatter(theta, act_pattern, c=control_weights, alpha=1.0, s=0.5, vmin=-vmax, vmax=vmax, cmap='coolwarm')

    vels = np.hstack((-config.vels[::-1], config.vels))
    vel_id = np.where(vels == v)[0][0]

    w_samples = [-150, -100, -50, 0, 50, 100, 150]
    normalize = Normalize(vmin=-vmax, vmax=vmax)
    scalarMap = ScalarMappable(norm=normalize, cmap="coolwarm")

    for w1 in w_samples:
        w1_ind = np.argmin(np.abs(w1-w))
        ax.axhline(s_bar[vel_id, w1_ind], zorder=2, linewidth=0.75, color=scalarMap.to_rgba(w1), alpha=1)

        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        axis_height_in_inches = bbox.height

        y_data_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        data_units_per_point = y_data_range / axis_height_in_inches / fig.dpi
        offset = (0.75 + 0.5) * data_units_per_point / 2

        y = s_bar[vel_id, w1_ind]
        ax.axhline(y - offset, zorder=1, linewidth=0.5, color='k', ls='-', alpha=1)
        ax.axhline(y + offset, zorder=1, linewidth=0.5, color='k', ls='-', alpha=1)


    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], labels=[r'$-\pi$', r'$-\frac{\pi}{2}$', '$0$', r'$\frac{\pi}{2}$', r'$\pi$']);
    ax.set_title('v = {:.0f} [deg / s]'.format(v))
    ax.set(xlabel=r'$\theta$ [rad]', ylabel='')
    ax.set_ylabel(r'$s$ [a.u.]')
    ax.set_ylim(0, 1.1*act_pattern.max())

    cax = fig.add_axes([0.775, 0.4, 0.015, 0.475])
    cb=plt.colorbar(im, cax=cax)
    cb.set_label(r'$w$')
    cb.set_ticks([-3 * config.sigma, 0, 3 * config.sigma])
    cb.set_ticklabels([r'$-3\sigma$', r'$0$', r'$3\sigma$'])

    plt.savefig(os.path.join(fig_path, fname), dpi=300)
    plt.close()


def plot_eigenvalue(vels, max_eigenvalues, fig_path, fname='max_eigenvalues.pdf'):
    
    fig, ax = plt.subplots(figsize=(1.5,0.9))
    ax.plot(vels, max_eigenvalues)
    ax.set_ylabel(r'$\max{\,(\Re{\,(\mathrm{eigenvalue})})}$')
    ax.set_xticks([-1800, -900, 0, 900, 1800])
    ax.set_xlim(-1800,1800)
    ax.set_xlabel('Target velocity [deg / s]')
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.yaxis.get_major_formatter().set_powerlimits((0, 0))  # Forces scientific notation
    plt.savefig(os.path.join(fig_path, fname), format='pdf', bbox_inches='tight', dpi=300)
    plt.close()


def steady_state_sim(training_set, alpha, k, fig_path):

    main_path = utils.get_data_path(config.data_path, config.params)
        
    data_path1 = os.path.join(main_path, 'sim={}'.format(k))
    control_weights = np.load(os.path.join(data_path1, 'control_weights.npy')).squeeze()
    feedback_weights = np.load(os.path.join(data_path1, 'feedback_weights.npy'))
    theta = np.load(os.path.join(data_path1, 'theta.npy'))

    data_path2 = os.path.join(data_path1, 'set={:d}_alpha={:.0e}'.format(training_set, alpha))
    #s = np.load(os.path.join(data_path2, 's_{}.npy').format(sim_type))
    readout_weights = np.load(os.path.join(data_path2, 'readout_weights.npy'))

    G = utils.nonlin_str2fun(config.nonlinearity)

    W_rec_sim = feedback_weights.T @ readout_weights + config.J0 / config.N
    dG = utils.d_nonlin_str2fun(config.nonlinearity)

    vels = np.hstack((-config.vels[::-1], config.vels))
    n_vels = len(vels)

    # loc_id = -1
    # act_patterns = np.zeros((n_vels, config.N))
    # for m in range(n_vels):
    #     act_patterns[m] = s[loc_id,m]

    network_params = utils.network_params_str(config.params)

    mu_theo = np.load(os.path.join('theory', network_params, 'closed_loop', 'mu.npy'))
    act_patterns = G(config.J0 * mu_theo[:,np.newaxis] + config.b + control_weights.T * config.vels_rad_per_ms[:,np.newaxis])

    # def printx(x):
    #    print(loss_fun(x))
    #d_loss_fun = lambda x: d_loss(x, 0, feedback_weights, readout_weights, control_weights, G, dG)


    mu_sim = np.mean(act_patterns, axis=-1)
    homogeneous_solution_num = np.zeros((n_vels, config.N))
    max_eigenvalues = []

    for i, test_vel in enumerate(vels):

        #vel_id = np.where(vels == test_vel)[0][0]
        #act_pattern = act_patterns[vel_id]

        v_in = config.vels_rad_per_ms[i]
        #inputs_theo = mu_theo[i] + config.b + control_weights.squeeze() * v_in

        loss_fun = lambda x: loss(x, v_in, feedback_weights, readout_weights, control_weights, G)
        res = minimize(loss_fun, act_patterns[n_vels//2].squeeze(), tol=1e-6, options={'disp': False})  #callback=printx
        steady_state = res.x

        homogeneous_solution_num[i] = steady_state

        plot_pattern(theta, control_weights, steady_state, test_vel, fig_path, network_params)

        x = W_rec_sim @ steady_state
        max_eigval = comp_max_eigval(x, W_rec_sim, control_weights, v_in, dG)
        max_eigenvalues.append(max_eigval)

    max_eigenvalues = np.stack(max_eigenvalues)

    plot_eigenvalue(vels, max_eigenvalues, fig_path)

    main_path = utils.get_data_path(config.data_path, config.params)
    data_path = os.path.join(main_path, 'sim={}'.format(k))
    data_path = os.path.join(data_path, 'set={}_alpha={}'.format(training_set, alpha))
    data_path = os.path.join(data_path, 'closed_loop')
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    np.save(os.path.join(data_path, 'max_eigenvalues.npy'), max_eigenvalues)
    np.save(os.path.join(data_path, 'homogeneous_state.npy'), homogeneous_solution_num)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='stability analysis')
    parser.add_argument('--set', type=int, required=False, default=0, help='set of training velocities')
    parser.add_argument('--alpha', type=float, required=False, default=1e-4, help='regularization parameter')
    args = parser.parse_args()

    network_params = utils.network_params_str(config.params)
    out_path = os.path.join('theory', network_params, 'stability')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    fig_path = os.path.join('figs', network_params, 'stability')
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    for k in range(config.n_sim):
        fig_path_sim = os.path.join(fig_path, 'sim={:d}'.format(k))

        if not os.path.exists(fig_path_sim):
            os.makedirs(fig_path_sim)

        steady_state_sim(args.set, args.alpha, k, fig_path_sim)
