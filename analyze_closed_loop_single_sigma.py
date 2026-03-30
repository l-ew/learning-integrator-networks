import numpy as np
import os
import matplotlib.pyplot as plt
import visualization
import pandas as pd
import seaborn as sns
import matplotlib_config
import colorcet as cc
from matplotlib import cm
from scipy.stats import binned_statistic
import argparse
import config
import utils
import re


def load_data(training_set, alpha, k, sim_type, params):

    main_path = utils.get_data_path(config.data_path, params)
    data_path1 = os.path.join(main_path, 'sim={}'.format(k))

    if not params['online']:
        data_path2 = os.path.join(data_path1, 'set={:d}_alpha={:.0e}'.format(training_set, alpha))
    
        with open(os.path.join(data_path2, 'fit_successful.txt'), 'r') as f:
            fit_successful = int(f.read())

    else:
        data_path2 = data_path1
        fit_successful = True

    if fit_successful:
        t = np.load(os.path.join(data_path2, 't_{}.npy'.format(sim_type)))
        s = np.load(os.path.join(data_path2, 's_{}.npy'.format(sim_type)), mmap_mode='r')
        psi = np.load(os.path.join(data_path2, 'psi_{}.npy'.format(sim_type)))
        theta = np.load(os.path.join(data_path1, 'theta.npy'))
        control_weights = np.load(os.path.join(data_path1, 'control_weights.npy'))

        if sim_type == 'velocity_trace':
            v = np.load(os.path.join(data_path2, '{}.npy').format(sim_type))
        else:
            v = None

        return theta, t, s, psi, control_weights, v


def load_avg_vels(training_set, alpha, k, sim_type, params):
    main_path = utils.get_data_path(config.data_path, params)

    if params['online']:
        data_path = main_path
    else:
        data_path = os.path.join(main_path, 'sim={}'.format(k))

    data_path = os.path.join(data_path, 'set={:d}_alpha={:.0e}'.format(training_set, alpha))
    avg_vels = np.load(os.path.join(data_path, 'avg_vels.npy'))
    return avg_vels


def plot_activity_multipe_time_points(path, training_set, alpha, params, vels, sim, fig_path):

    main_path = utils.get_data_path(path, params)

    data_path = os.path.join(main_path, 'sim={}'.format(sim))
    s_driven_final = np.load(os.path.join(data_path, 's_driven_final.npy'))

    data_path = os.path.join(main_path, 'sim={}'.format(sim), 'set={:d}_alpha={:.0e}'.format(training_set, alpha))
    #s_vel = np.load(os.path.join(data_path, 's_vels.npy'))

    perturbation = 0
    s_vel = np.load(os.path.join(data_path, 'perturbations', 's_{}.npy'.format(perturbation)))
    
    data_path = os.path.join(main_path, 'sim={}'.format(sim))
    theta = np.load(os.path.join(data_path, 'theta.npy'))

    k = np.where(vels==120)[0][0]

    s = s_vel[:,k,:].squeeze()
    time_points = np.array([0, 100, 200, 500, 1000]) / config.dt
    time_points = time_points.astype(int)

    import matplotlib as mpl
    cmap = mpl.colormaps['copper_r']
    colors = cmap(np.linspace(0,1,len(time_points)))

    fig = plt.figure(figsize=(2.25,1.1))
    ax = fig.add_axes([0.15, 0.35, 0.55, 0.6]) 

    im = ax.scatter(theta, s_driven_final[k], marker='.', edgecolors='none', color='red', s=1.0, alpha=1.0)
    for t, c in zip(time_points, colors):
        ax.scatter(theta, s[t], marker='.', edgecolors='none', color=c, s=1.0, alpha=1.0, label=r'$t$ = {} s'.format(t*config.dt/1000))

    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], labels=[-180, -90, 0, 90, 180]);
    ax.set(xlabel=r'$\theta$ [deg]', ylabel='')
    ax.set_ylabel('Activation [a.u.]')
    lgnd = plt.legend(bbox_to_anchor=(1, 1))
    for handle in lgnd.legend_handles:
        handle.set_sizes([15])

    plt.savefig(os.path.join(fig_path, 'activity_multipe_time_points.pdf'), dpi=300)
    plt.close()


def compare_cl_to_driven(path, training_set, alpha, params, vels, sim, fig_path):

    # v = np.tile(vels, n_sim)
    # rmse = []
    # t = []
 
    main_path = utils.get_data_path(path, params)
    
    # for j in range(n_sim):
    #     data_path = os.path.join(main_path, 'sim={}'.format(j), 'set={:d}_alpha={:.0e}'.format(training_set, alpha))
    #     rmse_cl_driven = np.load(os.path.join(data_path, 'normalized_rmse_cl_driven.npy'))
    #     rmse.append(rmse_cl_driven)

    # only use sim=0

    # rmse = np.hstack(rmse)

    # data = {'v': v, 'rmse': 100 * rmse}
    # df = pd.DataFrame(data)

    df = pd.read_csv(os.path.join(main_path, 'sim={}'.format(sim), 'set={:d}_alpha={:.0e}'.format(training_set, alpha), 'normalized_rmse_cl_driven.csv'))

    import matplotlib as mpl
    cmap = mpl.colormaps['copper_r']

    t = df['t [s]'].unique()
    colors = cmap(np.linspace(0,1,len(t)))
    palette = dict(zip(t, colors))

    #fig = plt.figure(figsize=(1.75,1.0))
    fig = plt.figure(figsize=(1.8,1.0))
    ax = sns.lineplot(x="v", y="rmse", hue='t [s]', errorbar='sd', palette=palette, marker=None, data=df);
    plt.grid()
    ax.set_xlim(-1800,1800)
    ax.set_xticks([-1800, -900, 0, 900, 1800])
    ax.legend(title=r'$t$ [s]')
    plt.xlabel('Velocity [deg/s]')
    plt.ylabel('Norm. RMSE [%]')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.savefig(os.path.join(fig_path, 'norm_rmse_cl_driven.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_pop_act_over_time(sim_type, conditions, t, s, theta, fig_path, double_ring=False):

    #ordering = np.argsort(theta)

    if sim_type == 'memory' or t[-1] > 60* 1000:
        xticks = 60 * np.arange(t[-1] // (60 * 1000) + 1)
        minutes = True
    elif sim_type == 'velocity_trace':
        xticks = [0,10,20,30]
        minutes = False
    else:
        xticks = None
        minutes = False

    for k, condition in enumerate(conditions):
        path = os.path.join(fig_path, sim_type)
        if not os.path.exists(path):
            os.makedirs(path)
        if sim_type == 'memory':
            fname = 'loc={}.pdf'.format(k)
        elif sim_type == 'vels':
            dt = config.dt
            fname = 'vel={}.pdf'.format(condition)
        else:
            dt = config.dt
            fname = '{}.pdf'.format(condition)

        if s.ndim == 3:
            act = s[:,k, :].squeeze()
            #plot_act_over_time(act, sim, path, dt=dt, skip=10, minutes=minutes)
            visualization.plot_act_over_time(t, act, path, fname=fname, xticks=xticks, minutes=minutes, double_ring=double_ring)
        else:
            visualization.plot_act_over_time(t, s, path, fname=fname, xticks=xticks, minutes=minutes, double_ring=double_ring)


def plot_tuning_curves(s, psi, fig_path):

    N = s.shape[0]
    n_sample = 10
    cell_ids = np.arange(n_sample) * N // n_sample

    fig, ax = plt.subplots(1, 1, figsize=(2.5,1.75), tight_layout=True)
    for i in range(n_sample):
        x = psi.squeeze()
        y = s[cell_ids[i],:].squeeze()
        x = np.hstack((x[-1]-2*np.pi, x, x[0]+2*np.pi))
        y = np.hstack((y[-1], y, y[0]))
        ax.plot(x, y, label='Cell # {}'.format(cell_ids[i]))
    #ax.legend(bbox_to_anchor=(1.04,1), borderaxespad=0, fontsize=15)
    #ax.set_ylim(0, 1.1 * s[cell_ids,:].nanmax())
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels([-180, -90, 0, 90, 180])
    ax.set_ylabel('Activity [a.u.]')
    ax.set_xlabel(r'$\psi$ [deg]')

    path = os.path.join(fig_path, 'tuning_curves.pdf')
    plt.savefig(path, format='pdf', dpi=300)
    plt.close()


def plot_theo_and_readout_psi_over_time(t, psi_theo, psi_readout, fig_path, xticks=None, labels=None, alignment_index=100):

    if labels is None:
        labels = [r'$\psi$', r'$\hat\psi$']

    fig = plt.figure(figsize=(2.25,1.1))
    ax = fig.add_axes([0.225, 0.35, 0.55, 0.6])

    if alignment_index is not None:
        psi_readout += psi_theo[alignment_index] - psi_readout[alignment_index]

    psi_theo = np.angle(np.exp(1j * psi_theo))
    psi_readout = np.angle(np.exp(1j * psi_readout))

    psi_theo[np.abs(np.diff(psi_theo,axis=0,append=np.nan))>np.pi] = np.nan
    psi_readout[np.abs(np.diff(psi_readout,axis=0,append=np.nan))>np.pi] = np.nan

    ax.plot(t / 1000, psi_theo, '--', color='k', linewidth=0.6, label=labels[0], zorder=1);
    ax.plot(t / 1000, psi_readout, color='orange', label=labels[1], zorder=0);

    leg = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    ax.set_xlabel(r'$t$ [s]')
    ax.set_ylabel('Angle [deg]')
    ax.set_xlim(0,np.max(t / 1000))

    if xticks is not None:
        ax.set_xticks(xticks)

    ax.set_ylim(-np.pi, np.pi)
    ax.set_yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi])
    ax.set_yticklabels([-180, -90, 0, 90, 180])

    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    plt.savefig(os.path.join(fig_path, 'psi_over_time.pdf'), dpi=300, bbox_extra_artists=(leg,), bbox_inches='tight')


def plot_psi_over_time_memory(t, psi, fig_path, fname='psi_over_time_memory.pdf', clusters=None, minutes=True):
    batch_size = psi.shape[1]

    psi[np.abs(np.diff(psi,axis=0,append=np.nan))>np.pi] = np.nan

    fig = plt.figure(figsize=(2.25,1.1))
    ax = fig.add_axes([0.225, 0.35, 0.55, 0.6]) 

    cmap = cc.cm.cyclic_isoluminant #cm.get_cmap('hsv', M)(np.arange(M))
    for i in range(batch_size):
        ax.plot(t/1000, psi[:,i], color=cmap((psi[0,i] + np.pi) / (2 * np.pi)), lw=0.5);

    ax.set_ylim(-np.pi, np.pi)
    ax.set_yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi], labels=[-180, -90, 0, 90, 180]);
    ax.set_ylabel(r'$\hat \psi$ [deg]')

    if clusters is not None:
        plt.scatter(len(clusters) * [t[-1]/1000], clusters)

    if minutes:
        n_min = int(np.round((t[-1] - t[0]) / (60 * 1000)))
        ax.set_xticks(60 * np.arange(n_min+1), labels=np.arange(n_min+1))
        ax.set_xlabel(r'$t$ [min]')
        ax.set_xlim(0, 60 * n_min)
    else:
        ax.set_xlabel(r'$t$ [s]')
        ax.set_xlim(0,t[-1]/1000)

    import matplotlib.colors as mcolors
    import matplotlib.colorbar as colorbar

    norm = mcolors.Normalize(vmin=-np.pi, vmax=np.pi)

    cbar = colorbar.ColorbarBase(fig.add_axes([0.8, 0.35, 0.015, 0.6]), cmap=cmap, norm=norm, orientation='vertical', ticks=[-np.pi, 0, np.pi])
    cbar.set_ticklabels([-180, 0, 180])
    cbar.set_label(r'$\hat \psi_0$ [deg]')

    path = os.path.join(fig_path, 'drift')
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(os.path.join(path, fname), format='pdf', dpi=300)


def scatter_psi_final_vs_psi_initial(psi_init, psi_final, fig_path, fname='scatter_psi_memory.pdf'):

    fig = plt.figure(figsize=(1.5,1.1))
    ax = fig.add_axes([0.25, 0.35, 0.55, 0.6]) 

    ax.scatter(psi_init, psi_final, s=2)
    ax.axis('square')
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi], labels=[-180, -90, 0, 90, 180]);
    ax.set_xlabel(r'Initial $\hat \psi$ [deg]')
    ax.set_ylim(-np.pi, np.pi)
    ax.set_yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi], labels=[-180, -90, 0, 90, 180]);
    ax.set_ylabel(r'Final $\hat \psi$ [deg]')

    path = os.path.join(fig_path, 'drift')
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(os.path.join(path, fname), format='pdf', dpi=300)


def cw_ccw_tuning_curve(psi, v, s, skip=10):
    cw_mask = v[::skip] * 180 * 1000 / np.pi < -90
    ccw_mask = v[::skip] * 180 * 1000 / np.pi > 90

    from scipy.stats import binned_statistic

    N = s.shape[1]
    bins = np.linspace(-np.pi, np.pi, 101, endpoint=True)
    cw_act = [s[cw_mask,k].squeeze() for k in range(N)]
    ccw_act = [s[ccw_mask,k].squeeze() for k in range(N)]

    cw_hd = psi[::skip][cw_mask].squeeze()
    cw_tuning, _, _ = binned_statistic(cw_hd, cw_act, statistic='mean', bins=bins)
    
    ccw_hd = psi[::skip][ccw_mask].squeeze()
    ccw_tuning, _, _ = binned_statistic(ccw_hd, ccw_act, statistic='mean', bins=bins)

    bin_means = bins[:-1] + 0.5 * (bins[1] - bins[0])

    return cw_tuning, ccw_tuning, bin_means



def plot_pattern(vels, theta, control_weights, s, fig_path, network_params):

    theo_path = os.path.join('theory', network_params, 'closed_loop')
    w = np.load(os.path.join(theo_path, 'w_values.npy'))
    s_bar = np.load(os.path.join(theo_path, 's.npy'))

    path2 = os.path.join('data', network_params, 'sim=0')
    turn_ind = np.load(os.path.join(path2, 'turn_ind.npy'))
    skip = 10
    n_steps = np.min(turn_ind) // skip

    vels = np.hstack((-config.vels[::-1], config.vels))
    n_vels = len(vels)

    loc_id = -1

    act_patterns = np.zeros((n_vels, config.N))
    for m in range(n_vels):
        act_patterns[m] = s[loc_id,m]

    plot_vel = 1500
    vel_id = np.where(vels == plot_vel)[0][0]
    act_pattern = act_patterns[vel_id]

    fname = 'pop_act_v={}_ind={}.pdf'.format(plot_vel,loc_id)

    fig, ax = plt.subplots(1, 1, figsize=(5,3))
    fig.tight_layout()

    vmax = np.maximum(np.max(control_weights), -np.min(control_weights))
    ax.scatter(theta, act_pattern, c=control_weights, alpha=1.0, s=5, vmin=-vmax, vmax=vmax, cmap='coolwarm')


    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    w_samples = [-100, -50, -25, 0, 25, 50, 100]
    normalize = Normalize(vmin=-vmax, vmax=vmax)
    scalarMap = ScalarMappable(norm=normalize, cmap="coolwarm")

    for w1 in w_samples:
        w1_ind = np.argmin(np.abs(w1-w))
        ax.hlines(s_bar[vel_id, w1_ind], xmin=-np.pi, xmax=np.pi, zorder=1, linewidth=1, color=scalarMap.to_rgba(w1), alpha=1)

    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], labels=[-180, -90, 0, 90, 180]);
    ax.set_title('v = {:.0f} [deg/s]'.format(plot_vel))
    ax.set(xlabel=r'$\theta$ [deg]', ylabel='')
    ax.set_ylabel('Activity [a.u.]')
    ax.vlines(0, ymin=0, ymax=1.1*act_pattern.max(), color='k', linestyle='--', alpha=0.5)
    ax.set_ylim(0, 1.1*act_pattern.max())

    plt.savefig(os.path.join(fig_path, fname), dpi=300, bbox_inches='tight')
    plt.close()


def comp_drift_fields(t_vels, psi_vels, vels, avg_vels):

    psi = []
    drift = []
    psi_bins = np.linspace(-np.pi, np.pi, 101, endpoint=True)
    psi_centers = psi_bins[:-1] + 0.5 * (psi_bins[1] - psi_bins[0])

    for p, v, mean_vel in zip(psi_vels.T, vels, avg_vels):
        k = extract_rotation_index(p, v)
        mean_vel *= np.pi / (180 * 1000.)
        #mean_vel = (p[-1] - p[k]) / (t_vels[-1] - t_vels[k])

        int_drift = p[k:] - mean_vel * t_vels[k:]

        from scipy.ndimage import uniform_filter1d
        int_drift = uniform_filter1d(int_drift, size=20)

        drift_vel = np.diff(int_drift) / np.diff(t_vels[k:])
        drift_vel *= 180 / np.pi * 1000

        psi_wrapped = np.angle(np.exp(1j * p[k:-1]))

        mean_drift = binned_statistic(psi_wrapped.flatten(), drift_vel.flatten(), statistic='mean', bins=psi_bins)[0]

        #sorting = np.argsort(psi_wrapped)

        #psi.append(psi_wrapped[sorting])

        psi.append(psi_centers)
        drift.append(mean_drift)

        #drift.append(drift_vel[sorting])

    return psi, drift


def dx_dt(t, x):
    y = np.diff(x, axis=0) / np.diff(t)[:,np.newaxis]
    return 180 / np.pi * 1000 * y


def comp_drift_field_memory(psi_memory, drift_memory):

    psi_bins = np.linspace(-np.pi, np.pi, 101, endpoint=True)
    psi_centers = psi_bins[:-1] + 0.5 * (psi_bins[1] - psi_bins[0])
    psi_memory = np.array([psi_memory[:-1,k] for k in range(psi_memory.shape[1])])
    drift_memory = np.array([drift_memory[:,k] for k in range(drift_memory.shape[1])])
    mean_drift_memory = binned_statistic(psi_memory.flatten(), drift_memory.flatten(), statistic='mean', bins=psi_bins)[0]
    #sd_drift_memory = binned_statistic(psi_memory.flatten(), drift_memory.flatten(), statistic='std', bins=psi_bins)[0]

    return psi_centers, mean_drift_memory

def extract_rotation_index(psi, v):
    x = np.sign(v) * psi[-1] - 2 * np.pi
    k = np.where(np.sign(v) * psi > x)[0][0]
    return k

def plot_drift_field(psi, drift, fig_path, fname='drift_field.pdf'):

    fig = plt.figure(figsize=(2.25,1.1))
    ax = fig.add_axes([0.225, 0.35, 0.55, 0.6]) 

    ax.plot(psi, drift)
    ax.axhline(0, linestyle='--', color='k', zorder=-1)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi], labels=[-180, -90, 0, 90, 180]);
    ax.set_xlabel(r'$\hat \psi$ [deg]')
    ax.set_ylabel('Drift [deg/s]')

    path = os.path.join(fig_path, 'drift')
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(os.path.join(path, fname), format='pdf', dpi=300)


def plot_drift_fields(psi, drift, vels, fig_path, fname='drift_fields.pdf'):

    fig = plt.figure(figsize=(3.5,2.25))
    ax = fig.add_axes([0.2, 0.2, 0.65, 0.6]) 

    v_max = 360

    import matplotlib.colors as mcolors
    cmap = plt.cm.cividis
    norm = mcolors.Normalize(vmin=0, vmax=v_max)

    for p, d, v in zip(psi, drift, vels):
        if v > 0 and v <= v_max:
            ax.plot(p, d, c=cmap(norm(v)), label=v, lw=0.5)
        elif v == 0:
            ax.plot(p, d, c=cmap(norm(v)), label=v, lw=1.0)

            crossings = np.where(np.diff(np.sign(d)) < 0)[0]

            print(crossings)
            print(p[:-1][crossings])

            ax.scatter(p[:-1][crossings] + 0.5 * (p[1] - p[0]), [0] * len(crossings), marker='.', s=40, linewidths=0.5, edgecolors='w', c='k', zorder=9)

        #    ax.plot(p, d, c='k', label=v, lw=1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
    cbar.set_label('Target velocity\n [deg/s]')
    cbar.set_ticks([0, 180, v_max])

    #ax.axhline(0, linestyle='--', color='k', lw=0.5, zorder=20)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi], labels=[-180, -90, 0, 90, 180]);
    ax.set_xlabel(r'$\hat \psi$ [deg]')
    ax.set_ylabel('Drift [deg/s]')

    ax.spines[['right', 'top']].set_visible(False)

    path = os.path.join(fig_path, 'drift')
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(os.path.join(path, fname), format='pdf', dpi=100)


def comp_phase_shifts(t_vels, psi_vels, vels, avg_vels):

    psi = []
    shift = []
    psi_bins = np.linspace(-np.pi, np.pi, 101, endpoint=True)
    psi_centers = psi_bins[:-1] + 0.5 * (psi_bins[1] - psi_bins[0])

    for p, v, mean_vel in zip(psi_vels.T, vels, avg_vels):
        k = extract_rotation_index(p, v)
        mean_vel *= np.pi / (180 * 1000.)

        phase_shift = p[k:] - mean_vel * t_vels[k:]
        offset = np.angle(np.mean(np.exp(1j * phase_shift)))
        phase_shift = np.angle(np.exp(1j * (phase_shift - offset)))

        x = np.angle(np.exp(1j * p[k:].flatten()))
        mean_phase_shift = binned_statistic(x, phase_shift.flatten(), statistic='mean', bins=psi_bins)[0]

        psi.append(psi_centers)
        shift.append(mean_phase_shift)

    return psi, shift


def plot_phase_shift(psi, shift, fig_path, fname='phase_shift.pdf'):

    fig = plt.figure(figsize=(2.25,1.1))
    ax = fig.add_axes([0.225, 0.3, 0.55, 0.6]) 

    ax.plot(psi, shift)
    ax.axhline(0, linestyle='--', color='k', zorder=-1)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi], labels=[-180, -90, 0, 90, 180]);
    ax.set_xlabel(r'$\hat \psi$ [deg]')
    ax.set_ylabel('Phase shift [deg]')

    path = os.path.join(fig_path, 'phase_shift')
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(os.path.join(path, fname), format='pdf', dpi=300)


def create_plots(training_set, alpha, k=0):
    network_params = utils.network_params_str(config.params)

    if config.online:
        fig_path = os.path.join('figs', network_params, 'closed_loop', 'online_alpha={:.0e}'.format(alpha))
    else:
        fig_path = os.path.join('figs', network_params, 'closed_loop', 'set={}_alpha={:.0e}'.format(training_set, alpha))
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    dr = config.input_structure == 'double_ring'
    vels = np.hstack((-config.vels[::-1], config.vels))

    sim_type = 'velocity_trace'
    theta, t, s, psi, control_weights, v = load_data(training_set, alpha, k, sim_type, config.params)
    dt = 1.0
    psi_theo = np.cumsum(v * dt)
    plot_pop_act_over_time(sim_type, [sim_type], t, s, theta, fig_path, double_ring=dr)
    plot_theo_and_readout_psi_over_time(t[1:], psi_theo, psi[1:].squeeze(), os.path.join(fig_path, sim_type), alignment_index=0)
    #cw_tuning, ccw_tuning, hd_means = cw_ccw_tuning_curve(psi_theo, v, s[:-1])
    #plot_tuning_curves(cw_tuning, hd_means, os.path.join(fig_path, sim_type))
    #plot_tuning_curves(ccw_tuning, hd_means, os.path.join(fig_path, sim_type))


    sim_type = 'vels'
    theta, t_vels, s_vels, psi_vels, control_weights, _ = load_data(training_set, alpha, k, sim_type, config.params)
    #plot_pop_act_over_time(sim_type, vels, t_vels, s_vels, theta, fig_path)
    #plot_psi_over_time(vels, k, t_vels, psi_vels, config.dt, fig_path)
    #plot_pattern(vels, theta, control_weights, s_vels, fig_path, network_params) # look at activity at the end of the simulation

    avg_vels = load_avg_vels(training_set, alpha, k, sim_type, config.params)

    v_example = 120
    vel_ind = np.where(vels == v_example)[0][0]
    t_ind = np.where(t_vels > 12000)[0][0]
    amp = 50.
    psi_avg = t_vels[:t_ind] / 1000 * avg_vels[vel_ind] * np.pi / 180
    psi_sim = psi_vels[:t_ind,vel_ind].squeeze()
    offset = np.angle(np.mean(np.exp(1j * (psi_sim - psi_avg))))
    psi_avg += offset
    psi_amp = psi_avg + amp * (psi_sim - psi_avg)
    xticks = [0, 6, 12]
    labels = [r'$v_{\text{avg}} \, t$', r'$\hat \psi$']
    plot_theo_and_readout_psi_over_time(t_vels[:t_ind], psi_avg, psi_amp, os.path.join(fig_path, sim_type), labels=labels, xticks=xticks, alignment_index=None)


    sim_type = 'memory'
    theta, t_memory, s_memory, psi_memory, _, _ = load_data(training_set, alpha, k, sim_type, config.params)

    if training_set == 1 or config.input_structure == 'double_ring':
        plot_psi_over_time_memory(t_memory, psi_memory[:,::10], fig_path)
    else:
        plot_psi_over_time_memory(t_memory, psi_memory, fig_path)

    scatter_psi_final_vs_psi_initial(psi_memory[0], psi_memory[-1], fig_path)
    locs = np.arange(psi_memory.shape[1])
    plot_pop_act_over_time(sim_type, locs, t_memory, s_memory, theta, fig_path)
    #plot_tuning_curves(s, psi, fig_path)


    drift_memory = dx_dt(t_memory, psi_memory)
    psi_centers, mean_drift_memory = comp_drift_field_memory(psi_memory, drift_memory)
    plot_drift_field(psi_centers, mean_drift_memory, fig_path, fname='drift_field.pdf')

    psi, drift = comp_drift_fields(t_vels, psi_vels, vels, avg_vels)

    psi.append(psi_centers)
    drift.append(mean_drift_memory)
    vels0 = np.hstack((vels, 0))

    plot_drift_fields(psi, drift, vels0, fig_path, fname='drift_field_.pdf')

    psi, shift = comp_phase_shifts(t_vels, psi_vels, vels, avg_vels)

    v_example = 120
    vel_ind = np.where(vels == v_example)[0][0]
    plot_phase_shift(psi[vel_ind], shift[vel_ind], fig_path, fname='phase_shift.pdf')


    sim_type = 'memory'
    for k in range(config.n_sim):
        _, t_memory, s_memory, psi_memory, _, _ = load_data(training_set, alpha, k, sim_type, config.params)
        fname = 'psi_over_time_memory_sim={}.pdf'.format(k)

        delta = np.diff(psi_memory[-1])
        delta = np.angle(np.exp(1j * delta))
        cluster_id = [0]
        n = 0
        eps = 1e-3
        for x in delta:
            if np.abs(x) > eps:
                n += 1
            cluster_id.append(n)

        cluster_id = np.array(cluster_id)

        if np.abs(np.angle(np.exp(1j * (psi_memory[-1][-1] - psi_memory[-1][0])))) < eps:
            m = cluster_id[-1]
            cluster_id = [x if x != m else 0 for x in cluster_id]

        clusters = []
        for n in np.unique(cluster_id):
            mask = cluster_id == n
            cluster_mean = np.exp(1j * psi_memory[-1][mask])
            cluster_mean = np.angle(np.mean(cluster_mean))
            clusters.append(cluster_mean)

        plot_psi_over_time_memory(t_memory, psi_memory, fig_path, fname=fname, clusters=clusters)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='plot performance')
    parser.add_argument('--set', type=int, required=False, default=0, help='set of training velocities')
    parser.add_argument('--alpha', type=float, required=False, default=1e-4, help='regularization parameter')
    parser.add_argument('--vary', type=str, required=False, default='sigma')
    parser.add_argument('--vals', type=int, nargs='+', required=False, default=[50,100,150,200,250], help='varied values')
    args = parser.parse_args()

    sim = 0

    if not config.online:

        network_params = utils.network_params_str(config.params)
        fig_path = os.path.join('figs', network_params, 'closed_loop', 'set={}_alpha={:.0e}'.format(args.set, args.alpha))
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)

        vels = np.hstack((-config.vels[::-1], config.vels))
        compare_cl_to_driven(config.data_path, args.set, args.alpha, config.params, vels, sim, fig_path)
        plot_activity_multipe_time_points(config.data_path, args.set, args.alpha, config.params, vels, sim, fig_path)

    create_plots(args.set, args.alpha, k=sim)