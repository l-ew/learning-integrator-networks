import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib_config
import utils
import argparse
import config


def plot_readout_error(df, vels_theo, theo_radial_error, theo_angular_error, k, fname, fig_path, v_train=None):

    colors = ['tab:green', 'tab:purple'] #plt.rcParams['axes.prop_cycle'].by_key()['color']
    palette = {'Radial error': colors[0], 'Tangential error': colors[1]}
    
    fig, ax = plt.subplots(1,1, figsize=(1.75,1.2))
    if k == 0:
        ax.plot(vels_theo, theo_radial_error, c=colors[0], linewidth=0.75, alpha=1, zorder=1)
        sns.lineplot(ax=ax, data=df, x="v", y="error", hue="error type", palette=palette, errorbar='sd', marker='.', linestyle='', err_kws={"linewidth": 0.75}, err_style='bars', markersize=5)
    else:
        ax.plot(vels_theo, theo_radial_error, c=colors[0], linewidth=0.75, label=None, alpha=1, zorder=1)
        ax.plot(vels_theo, theo_angular_error, c=colors[1], linewidth=0.75, label=None, alpha=1, zorder=1)
        #sns.scatterplot(data=df, x="v", y="error", hue="error_type", size=20)
        sns.lineplot(ax=ax, data=df, x="v", y="error", hue="error type", palette=palette, errorbar='sd', marker='.', linewidth=0.5, linestyle='', markeredgewidth=0.5, markeredgecolor='k', err_kws={"linewidth": 0.75, "zorder": -1}, err_style='bars', markersize=5)
        ax.legend(bbox_to_anchor=(0.4, 1.1), loc='upper left');

    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.spines[['right', 'top']].set_visible(False)

    v_thr = 900
    mask = np.abs(df['v']) < v_thr
    min_error = np.min(df['error'][mask])
    max_error = np.max(df['error'][mask])
    ax.set_ylim(min_error, max_error)

    # ax.hlines(0, xmin=-v_thr, xmax=v_thr, color='k')
    # if v_train is not None:
    #     ax.vlines(-v_train, ymin=min_error, ymax=max_error, color='k')
    #     ax.vlines(v_train, ymin=min_error, ymax=max_error, color='k')
    ax.set_xlabel(r'$v$ [deg / s]')
    ax.set_ylabel('Error')
    ax.set_xlim(-v_thr, v_thr)
    ax.set_xticks(360 * np.arange(-2, 3))
    ax.grid('on')
    plt.savefig(os.path.join(fig_path, fname), dpi=300, bbox_inches='tight')
    plt.close()


def load_readout_errors(data_path):
    radial_errors = np.load(os.path.join(data_path, 'radial_errors.npy'))
    tangential_errors = np.load(os.path.join(data_path, 'tangential_errors.npy'))
    return radial_errors, tangential_errors


def main():
    parser = argparse.ArgumentParser(description='Compute macroscopic quantities.')
    parser.add_argument('--nsim', type=int, required=False, default=1, help='number of simulation runs')
    parser.add_argument('--alpha', type=float, required=False, default=1e-4, help='regularization parameter')
    parser.add_argument('--vel', type=int, required=False, default=360, help='training velocity')
    args = parser.parse_args()

    network_params = utils.network_params_str(config.params)
    fig_path_readout = os.path.join('figs', network_params, 'readout')
    os.makedirs(fig_path_readout, exist_ok=True)

    sim_path = os.path.join('simulation', network_params)
    theo_path = os.path.join('theory', network_params)

    theo_ro_path = os.path.join(theo_path, 'v_train={}'.format(args.vel))
    radial_errors_theo, tangential_errors_theo = load_readout_errors(theo_ro_path)
    vels_theo = np.load(os.path.join(theo_path, 'vels.npy'))

    data_path = utils.get_data_path(config.data_path, config.params)
    data_path = os.path.join(data_path, 'sim={}'.format(0))

    vels = np.load(os.path.join(data_path, 'vels.npy'))
    n_vels = len(vels)
    v = np.repeat(vels[np.newaxis,:], config.n_modes, axis=0)
    v = v.flatten()
    v = np.tile(v, args.nsim)

    modes = np.arange(config.n_modes)
    modes = np.repeat(modes[:,np.newaxis], n_vels, axis=1)
    modes = modes.flatten()
    modes = np.tile(modes, args.nsim)

    radial_errors = []
    tangential_errors = []
    simulation = []

    for sim in range(args.nsim):
        sim_ro_path = os.path.join(sim_path, 'sim={}'.format(sim), 'v_train={}'.format(args.vel))
        re, te = load_readout_errors(sim_ro_path)
        radial_errors.append(re.flatten())
        tangential_errors.append(te.flatten())
        simulation.append([sim] * len(re.flatten()))

    radial_errors = np.hstack(radial_errors)
    tangential_errors = np.hstack(tangential_errors)
    simulation = np.hstack(simulation)

    data = {'v': v, 'mode': modes, 'simulation': simulation, 'Radial error': radial_errors, 'Tangential error': tangential_errors}
    df = pd.DataFrame(data)

    df = pd.melt(df, id_vars=['v', 'mode', 'simulation'], value_vars=['Radial error', 'Tangential error'], var_name='error type', value_name='error')

    for k in range(1, config.n_modes):
        fname = 'readout_error_v_train={}_mode={}_aggregated.pdf'.format(args.vel, k)
        plot_readout_error(df[df['mode']==k], vels_theo, radial_errors_theo[k], tangential_errors_theo[k], k, fname, fig_path_readout, v_train=args.vel)


if __name__ == "__main__":
    main()