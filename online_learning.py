import numpy as np
import argparse
import network
import torch
import utils
import config
import os
from compare_readout_weights import plot_simulation_connectivity_continuum
from simulate_closed_loop import drop_unused_modes
from visualization import plot_bump_shapes_over_time
from plot_driven_network import plot_bump_shape
import visualization
import matplotlib_config


def psi2z(psi, n_modes):
    time_steps, n_vels = psi.shape
    z_target = torch.ones((time_steps, n_vels, 2*n_modes-1)).float()
    for k in range(1,n_modes):
        z_target[:,:,2*k-1] = torch.cos(k * psi)
        z_target[:,:,2*k] = torch.sin(k * psi)
    return z_target


def generate_velocity_trace_alternate(vel, time_steps, eps=0.1):
    v = vel * np.ones(time_steps) * np.pi / 180
    k = 0
    direction = 1
    while k < time_steps:
        x = int(360 * 1000 / (vel * config.dt))
        x += int(eps * x * (np.random.rand() - 0.5))

        if k + x < time_steps:
            v[k:k+x] *= direction

        k += x
        direction *= -1

    return v


def plot_proj_over_time(data, cmap='cool', aspect=None, ticks = [0, 100, 200, 300], figpath='', figname='trajectory_plot.pdf'):
    
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    time_steps, batch_size, dims = data.shape
    assert dims == 2, "Third dimension must be 2 (x,y coordinates)"
    
    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_axes([0.25, 0.25, 0.5, 0.65]) 
    
    norm = mpl.colors.Normalize(vmin=0, vmax=time_steps-1)
    color_map = plt.cm.get_cmap(cmap)
    
    for b in range(batch_size):
        trajectory = data[:, b, :]

        for t in range(time_steps-1):
            ax.plot(trajectory[t:t+2, 0], trajectory[t:t+2, 1], 
                    color=color_map(norm(t)), linewidth=0.5)
    
    sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
    sm.set_array([])

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(sm, cax=cax, label='time [s]')
    
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(["{:.1f}".format(t/1000) for t in ticks])
    cbar.set_label(r'$t$ [s]')

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    if aspect == 'equal':
        ax.set_xlim(-0.75, 0.75)
        ax.set_ylim(-0.75, 0.75)
        ax.set_aspect('equal')
        
    full_path = os.path.join(figpath, figname)
    plt.savefig(full_path, dpi=300)
    plt.close()


def plot_error_over_time(t, error, ylabel=None, figpath='', figname='error_over_time.pdf'):
    
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(2.5, 1.75))
    ax = fig.add_axes([0.25, 0.25, 0.65, 0.65]) 
    plt.plot(t, error)
    plt.xlim(0, t[-1])
    plt.xlabel(r'$t$ [s]')
    if ylabel is not None:
        plt.ylabel(ylabel)
    else:
        plt.ylabel('Error')
    plt.yscale('log')
    full_path = os.path.join(figpath, figname)
    plt.savefig(full_path, dpi=300)
    plt.close()


def plot_vel_over_time(t, input_vel, readout_vel, figpath='', figname='vel_over_time.pdf'):
    
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(2.5, 1.75))
    ax = fig.add_axes([0.25, 0.25, 0.65, 0.65]) 
    plt.plot(t, readout_vel)
    plt.plot(t, input_vel, 'k--')
    plt.xlim(0, t[-1])
    plt.ylim(-500,500)
    plt.xlabel(r'$t$ [s]')
    plt.ylabel('Velocity [deg/s]')
    full_path = os.path.join(figpath, figname)
    plt.savefig(full_path, dpi=300)
    plt.close()


def plot_errors_over_time(t, error1, error2, ylabel1=None, ylabel2=None, 
                          color1='blue', color2='red',
                          figpath='', figname='dual_errors_over_time.pdf'):

    # Handle potential log scale issues
    error1 = np.maximum(error1, 1e-16)
    error2 = np.maximum(error2, 1e-16)
    
    fig = plt.figure(figsize=(2.75, 1.75))  # Slightly larger to accommodate two y-axes
    
    # Create first axis (left)
    ax1 = fig.add_axes([0.2, 0.25, 0.65, 0.65])
    ax1.plot(t, error1, color=color1, linewidth=0.75, zorder=3)
    ax1.set_xlim(0, t[-1])
    ax1.set_xlabel(r'$t$ [s]')
    ax1.set_yscale('log')
    
    # Set left y-axis label and color
    if ylabel1 is not None:
        ax1.set_ylabel(ylabel1, color=color1)
    else:
        ax1.set_ylabel('Error 1', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Create second axis (right) sharing the same x-axis
    ax2 = ax1.twinx()
    ax2.plot(t, error2, color=color2, linewidth=0.75, zorder=0)
    ax2.set_yscale('log')

    ax1.spines[['top']].set_visible(False)
    ax2.spines[['top']].set_visible(False)

    # Set right y-axis label and color
    if ylabel2 is not None:
        ax2.set_ylabel(ylabel2, color=color2)
    else:
        ax2.set_ylabel('Error 2', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)  # Make ax1 background transparent

    # Save figure
    if figpath and not os.path.exists(figpath):
        os.makedirs(figpath)
    
    full_path = os.path.join(figpath, figname)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()


def online_learning(feedback_net, s, psi, vels, time_steps, flip_ind, params, device, alpha=1., zero_crossings=None, use_modes=None, path='.', figpath='.'):

    t = params['dt'] * torch.arange(time_steps)
    #psi = psi0.unsqueeze(0) + vels.unsqueeze(0) * t.unsqueeze(1)

    z_target = psi2z(psi, params['n_modes'])
    mask = utils.modes_mask(params['n_modes'], use_modes=use_modes)
    z_target = z_target[:,:,mask]
    z_readout = torch.zeros_like(z_target)

    M = vels.shape[1]
    zeta = 5.0e-6  #  1e-5 for exp + cos, 1e-7 for relu + cos # 1e-6
    out_weights = feedback_net.out_weights

    momentum = 0.0
    grad = 0

    k = 0
    zero_crossings = np.hstack([zero_crossings, np.inf])

    path_weights = os.path.join(path, 'intermediate_readout_weights')
    if not os.path.exists(path_weights):
        os.makedirs(path_weights)

    path_act = os.path.join(path, 'intermediate_activations')
    if not os.path.exists(path_act):
        os.makedirs(path_act)

    theta = np.linspace(-np.pi, np.pi, params['N'], endpoint=False)
    theta = torch.from_numpy(theta).float()
    proj_weights = torch.stack((torch.cos(theta), torch.sin(theta))).T / params['N']
    proj_weights = proj_weights.to(device)

    proj = torch.zeros((time_steps, 2))
    Deltas = torch.zeros((time_steps, 2))

    s_sim = torch.zeros((flip_ind[1] - flip_ind[0], params['N']))
    interval = 0

    for i in range(time_steps):

        s, out = feedback_net(s, vels[i].reshape(M,1).to(device))
        Delta = out - z_target[i].to(device)  # dim = M x n_modes
        grad = momentum * grad + (1 - momentum) * (torch.matmul(Delta.T, s))

        s_sim[i-flip_ind[interval]] = s.detach().cpu()
        proj[i] = (s @ proj_weights).detach().cpu()
        z_readout[i] = out.detach().cpu()
        Deltas[i] = Delta.detach().cpu()

        if i + 1 == flip_ind[interval+1]:

            t_sim = t[flip_ind[interval]:flip_ind[interval+1]] - t[flip_ind[interval]]
            t_sim = t_sim.numpy()

            xticks = 0.25 * np.arange(int(t_sim[-1] // 250) + 1)

            fname = 'act_over_time_interval={}.pdf'.format(interval)
            visualization.plot_act_over_time(t_sim, s_sim.numpy(), figpath, xticks=xticks, fname=fname, double_ring=False)

            interval += 1

            if flip_ind[interval+1] != -1:
                s_sim = torch.zeros((flip_ind[interval+1] - flip_ind[interval], params['N']))


        if i == zero_crossings[k]:

            np.save(os.path.join(path_weights, 'readout_weights_i={}.npy'.format(i)), out_weights.detach().cpu().numpy())
            np.save(os.path.join(path_act, 's_i={}.npy'.format(i)), s.detach().cpu().numpy())
            k += 1

        if i > 2000:
            out_weights -= zeta * (grad + alpha * out_weights)

    return s.detach().cpu().numpy(), psi[-1].detach().cpu().numpy(), out_weights.detach().cpu().numpy(), z_readout.numpy(), Deltas.numpy(), proj.numpy()


def window_average(t, data, window_size=1000):
    t_mean = []
    data_mean = []
    for start in range(0, len(t), window_size):
        data_mean.append(np.mean(data[start:start+window_size]))
        t_mean.append(np.mean(t[start:start+window_size]))
    return np.array(t_mean), np.array(data_mean)

def run_sim(device, params, alpha=1, save_traces=False, use_modes=None, sim=0):

    path = utils.get_data_path(config.data_path, params)
    path = os.path.join(path, 'sim={}'.format(sim))
    if not os.path.exists(path):
        os.makedirs(path)

    network_params = utils.network_params_str(config.params)
    fig_path = os.path.join('figs', network_params, 'closed_loop', 'online_alpha={:.0e}'.format(alpha))
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    fig_path_connectivity = os.path.join(fig_path, 'connectivity')
    if not os.path.exists(fig_path_connectivity):
        os.makedirs(fig_path_connectivity)

    fig_path_activity = os.path.join(fig_path, 'activity')
    if not os.path.exists(fig_path_activity):
        os.makedirs(fig_path_activity)

    fig_path_proj = os.path.join(fig_path, 'proj')
    if not os.path.exists(fig_path_proj):
        os.makedirs(fig_path_proj)

    np.random.seed(8245)

    init_offline = False
    if init_offline:

        path1 = utils.get_data_path(config.data_path, params)
        path1 = path1.replace('online=True', 'online=False')
        path2 = os.path.join(path1, 'sim=0')

        feedback_weights = np.load(os.path.join(path2, 'feedback_weights.npy'))
        control_weights = np.load(os.path.join(path2, 'control_weights.npy'))
        theta = np.load(os.path.join(path2, 'theta.npy'))

        out_weights = np.load(os.path.join(path2, 'set=0_alpha=1e-04', 'readout_weights.npy'))
        out_weights = drop_unused_modes(out_weights, use_modes)

        #out_weights += 0.25 * np.sqrt(np.mean(out_weights**2)) * np.random.randn(2 * config.N).reshape(2,config.N)

        # out_weights = np.zeros(2 * config.N).reshape(2,config.N)
        # out_weights[0,:] = np.cos(theta) / config.N
        # out_weights[1,:] = np.sin(theta) / config.N

        feedback_weights = drop_unused_modes(feedback_weights, use_modes)

    else:

        feedback_weights, control_weights, theta = network.generate_weights(params)
        feedback_weights = drop_unused_modes(feedback_weights, use_modes)

        #out_weights = 1.0 / params['N'] * np.random.randn(np.prod(feedback_weights.shape)).reshape(feedback_weights.shape)
        out_weights = np.zeros(np.prod(feedback_weights.shape)).reshape(feedback_weights.shape)


    driven_net, in_weights = network.init_driven_network(control_weights, feedback_weights, params, device)
    feedback_net = network.init_feedback_network(control_weights, out_weights, feedback_weights, params, device)

    v_train = 360
    M = 1
    psi0 = -np.pi * np.ones(1)
    time_steps = 300000  # note: use dt = 1.0
    t = np.arange(time_steps) / 1000 * params['dt']

    v = generate_velocity_trace_alternate(v_train, time_steps)

    #v = utils.generate_velocity_trace(time_steps)

    print(np.mean(np.abs(v * 180 / np.pi)))

    vel_trace = v / 1000

    control_inputs = torch.from_numpy(vel_trace).float()
    control_inputs = control_inputs.reshape(len(control_inputs),1)

    psi = psi0 + np.cumsum(vel_trace * params['dt'])
    psi = psi.reshape(len(psi),1)

    T = (2 * np.pi / np.mean(np.abs(vel_trace)))
    alpha_scaled = alpha / (T / params['dt'])

    zero_crossings = np.where(np.diff(np.sign(psi.squeeze())))[0]
    zero_crossings = np.hstack([0, zero_crossings])
    vel_sign_crossings = [1 if v[k] > 0 else -1 for k in zero_crossings]

    np.savetxt(os.path.join(path, "zero_crossings.csv"), zero_crossings, fmt='%i', delimiter="\n")
    np.savetxt(os.path.join(path, "vel_sign_zero_crossings.csv"), vel_sign_crossings, fmt='%i', delimiter="\n")

    np.save(os.path.join(path, 't_train.npy'), t)
    np.save(os.path.join(path, 'v_train.npy'), v)
    np.save(os.path.join(path, 'psi_train.npy'), psi)

    s0 = torch.zeros(M, params['N']).float().to(device)
    psi = torch.from_numpy(psi).float().to(device)

    flip_ind = np.where(np.abs(np.diff(np.sign(vel_trace))) > 0)[0]
    flip_ind = np.hstack((0, flip_ind, len(vel_trace)+1))

    np.savetxt(os.path.join(path, "flip_ind.csv"), flip_ind, fmt='%i', delimiter="\n")

    s_final, psi_final, readout_weights, z_readout, delta, proj = online_learning(feedback_net, s0, psi, control_inputs, time_steps, flip_ind, params, device, zero_crossings=zero_crossings, alpha=alpha_scaled, use_modes=use_modes, path=path, figpath=fig_path_activity)

    np.save(os.path.join(path, 'feedback_weights.npy'), feedback_weights)
    np.save(os.path.join(path, 'readout_weights.npy'), readout_weights)
    np.save(os.path.join(path, 's_online_final.npy'), s_final)
    np.save(os.path.join(path, 'psi_online_final.npy'), psi_final)
    np.save(os.path.join(path, 'z_readout.npy'), z_readout)
    np.save(os.path.join(path, 'delta.npy'), delta)
    np.save(os.path.join(path, 'proj.npy'), proj)
    np.save(os.path.join(path, 'control_weights.npy'), control_weights)
    np.save(os.path.join(path, 'theta.npy'), theta)

    vels = np.hstack((-config.vels[::-1], config.vels))
    np.save(os.path.join(path, 'vels.npy'), vels)

    turn_ind = np.zeros(len(vels), dtype=int)
    for i, v in enumerate(config.vels_rad_per_ms):
        turn_ind[i] = int(np.round(np.abs(2 * np.pi / (params['dt'] * v))))
    np.save(os.path.join(path, 'turn_ind.npy'), turn_ind)

    n_std = 5
    w_grid = np.linspace(-n_std*config.sigma, n_std*config.sigma, 51)

    for i in zero_crossings:
        readout_weights = np.load(os.path.join(path, 'intermediate_readout_weights', 'readout_weights_i={}.npy'.format(i)))
        W_rec = feedback_weights.T @ readout_weights

        fname = 'connectivity_simulation_i={}.pdf'.format(i)
        plot_simulation_connectivity_continuum(w_grid, theta, config.sigma, control_weights, W_rec, fig_path_connectivity, fname=fname)

        s_i = np.load(os.path.join(path, 'intermediate_activations', 's_i={}.npy'.format(i)))
        fname='bump_shape_i={}.pdf'.format(i)
        plot_bump_shape(theta, s_i, v_train, fig_path_activity, w=control_weights, color_w=True, display_loc=False, ymax=4, fname=fname)

    for i, (k1, k2) in enumerate(zip(flip_ind[:-1], flip_ind[1:])):
        fname = 'proj_segment={}.pdf'.format(i)

        plot_proj_over_time(np.expand_dims(proj[k1:k2], 1), figname=fname, figpath=fig_path_proj)


    # s = []
    # for i in [zero_crossings[0], zero_crossings[4], zero_crossings[10], zero_crossings[20]]:
    #     s_i = np.load(os.path.join(path, 'intermediate_activations', 's_i={}.npy'.format(i)))
    #     s.append(s_i)

    # plot_bump_shapes_over_time(theta, s, fig_path)


def generate_error_plots(sim=0, alpha=0):

    path = utils.get_data_path(config.data_path, config.params)
    path = os.path.join(path, 'sim={}'.format(sim))

    network_params = utils.network_params_str(config.params)
    fig_path = os.path.join('figs', network_params, 'closed_loop', 'online_alpha={:.0e}'.format(alpha), 'error')
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    t = np.load(os.path.join(path, 't_train.npy'))
    vel_trace = np.load(os.path.join(path, 'v_train.npy')) / 1000
    delta = np.load(os.path.join(path, 'delta.npy'))
    z_readout = np.load(os.path.join(path, 'z_readout.npy'))

    error = np.sum(delta**2, axis=-1)
    t_error, mean_error = window_average(t, error, window_size=1000)
    mean_error = np.sqrt(mean_error)

    figname = 'readout_error_over_time.pdf'
    plot_error_over_time(t_error, mean_error, figpath=fig_path, ylabel='Readout error', figname=figname)

    psi_readout = np.angle(z_readout[:,0,0] + 1j * z_readout[:,0,1])
    psi_readout = np.unwrap(psi_readout)
    v_readout = np.diff(psi_readout) / config.params['dt']
    v_readout = np.hstack(([np.nan], v_readout))

    k = -3000
    plot_vel_over_time(t[k:]-t[k], vel_trace[k:] * 1000 * 180 / np.pi, v_readout[k:] * 1000 * 180 / np.pi, fig_path)

    velocity_error = np.abs(v_readout - vel_trace) * 1000 * 180 / np.pi
    #velocity_error = velocity_error**2
    t_error, mean_velocity_error = window_average(t, velocity_error, window_size=1000)
    #mean_velocity_error = np.sqrt(mean_velocity_error)

    fname = 'velocity_error_over_time.pdf'
    plot_error_over_time(t_error, mean_velocity_error, figpath=fig_path, ylabel='Velocity error [deg/s]', figname=fname)

    plot_errors_over_time(t_error, mean_error, mean_velocity_error, 
                              ylabel1='Readout error', ylabel2='Velocity error [deg/s]',
                              color1='k', color2='tab:blue', figpath=fig_path)


def main(alpha, save_traces, use_modes, sim):
    device = torch.device('cuda:0')
    params = config.params
    params['online'] = True

    run_sim(device, params, alpha=alpha, save_traces=save_traces, use_modes=use_modes, sim=sim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learn path integration networks.')
    parser.add_argument('--alpha', type=float, default=0, help='regularization parameter')
    parser.add_argument('--sim', type=int, default=0, help='simulation number')
    parser.add_argument('--modes', type=int, nargs='+', default=[1], required=False)
    parser.add_argument('--save-traces', dest='save_traces', action='store_true')
    parser.add_argument('--discard-traces', dest='save_traces', action='store_false')
    parser.set_defaults(save_traces=False)
    args = parser.parse_args()

    main(args.alpha, args.save_traces, args.modes, args.sim)
