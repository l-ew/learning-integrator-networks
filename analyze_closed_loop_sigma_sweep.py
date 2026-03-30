import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib_config
import argparse
import config
import utils
import re


def aggregate_linear_data(path, quantity, values, training_set, alpha, params, test_vels, n_sim, online=False):
    n_vels = len(test_vels)
    params = params.copy()
    df = pd.DataFrame(columns=[quantity, 'sim', 'target_vel', 'avg_vel', 'delta_vel', 'r2'])
    for i, x in enumerate(values):

        params[quantity] = x

        main_path = utils.get_data_path(path, params)
        
        for j in range(n_sim):
        
            if online:
                data_path = os.path.join(main_path, 'sim={}'.format(j))
            else:
                data_path = os.path.join(main_path, 'sim={}'.format(j), 'set={:d}_alpha={:.0e}'.format(training_set, alpha))
            
            if online:
                fit_successful = True
            else:
                with open(os.path.join(data_path, 'fit_successful.txt'), 'r') as f:
                    fit_successful = int(f.read())
            
            if fit_successful:
                avg_vels = np.load(os.path.join(data_path, 'avg_vels.npy'))
                r2_scores = np.load(os.path.join(data_path, 'r2_scores.npy'))
             
                for k in range(n_vels):
                    df.loc[len(df)] = [x, j, test_vels[k], avg_vels[k], avg_vels[k]-test_vels[k], r2_scores[k]]

    if quantity == 'sigma':
        df = df.astype({quantity:'int', 'sim':'int', 'target_vel':'int'})
    else:
        df = df.astype({'sim':'int', 'target_vel':'int'})
    return df


def plot_delta_vel(df, quantity, input_type, figpath, estimator='mean', fname='delta_vel', ylim=None, v_train=None):
    fig = plt.figure(figsize=(2.25,1.25))
    ax = fig.add_axes([0.225, 0.25, 0.45, 0.7])

    if df['sim'].max() > 1:
        ax2 = sns.lineplot(x="target_vel", y="delta_vel", hue=quantity, estimator=estimator, errorbar='sd', data=df, legend='full');
    else:
        ax2 = sns.lineplot(x="target_vel", y="delta_vel", hue=quantity, data=df, legend='full');

    leg=ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    #plt.grid()
    plt.xticks([-720, -360, 0, 360, 720])
    plt.xlim(-900,900)
    if quantity == 'sigma' and input_type == 'double_ring':
        leg.set_title(r'$u_0$')
    elif quantity == 'sigma' and input_type == 'random':
        leg.set_title(r'$\sigma$')
    else:
        leg.set_title('')
        #handles, labels = ax2.get_legend_handles_labels()
        #new_labels = [label[:3] for label in labels]
        #ax2.legend(handles, new_labels, title = r'$g$')

    if ylim is not None:
        plt.ylim(ylim)
    else:
        deltas = df['delta_vel'][df['target_vel'].abs() < 901]
        ylim = 1.05 * np.maximum(-np.min(deltas), np.max(deltas))
        plt.ylim(-ylim, ylim)

    ax.axhline(0, linewidth=0.5, linestyle='--', color='k')
    if v_train is not None:
        ax.axvline(-v_train, linewidth=0.5, linestyle='--', color='k', zorder=-1)
        ax.axvline(v_train, linewidth=0.5, linestyle='--', color='k', zorder=-1)

    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    plt.xlabel('Target velocity [deg/s]')
    plt.ylabel(r'$\Delta$ velocity [deg/s]')
    plt.savefig(os.path.join(figpath, '{}_{}.pdf'.format(fname, quantity)), dpi=300, bbox_extra_artists=(leg,))
    plt.close()


def plot_vel(df, quantity, input_type, figpath, estimator='mean'):
    fig = plt.figure(figsize=(2.0,1.35))
    ax = fig.add_axes([0.25, 0.3, 0.55, 0.65])

    ax2 = sns.lineplot(x="target_vel", y="avg_vel", hue=quantity, estimator=estimator, errorbar='sd', data=df, legend='full');

    leg=ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.xlabel('Target velocity [deg/s]')
    plt.ylabel('Bump velocity \n [deg/s]')
    plt.grid()
    #plt.xticks(np.arange(-3000, 4000, 1500))
    #plt.xlim(-3600,3600)
    plt.axis('square')
    ax2.set_xlim(-1800,1800)
    ax2.set_ylim(-1800,1800)
    ax2.set_xticks([-1800, -900, 0, 900, 1800])
    ax2.set_yticks([-1800, -900, 0, 900, 1800])
    plt.xticks(fontsize=5, rotation=30)
    ax.tick_params(axis='x', which='major', pad=3)

    if quantity == 'sigma' and input_type == 'double_ring':
        leg.set_title(r'$u_0$')
    elif quantity == 'sigma' and input_type == 'random':
        leg.set_title(r'$\sigma$')
    else:
        leg.set_title(r'$g$')
        handles, labels = ax2.get_legend_handles_labels()
        new_labels = [label[:3] for label in labels]
        ax2.legend(handles, new_labels, title = r'$g$')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    plt.savefig(os.path.join(figpath, 'vel_{}.pdf'.format(quantity)), dpi=300, bbox_extra_artists=(leg,))
    plt.close()


def plot_R2_vel(df, quantity, input_type, figpath, estimator='mean'):
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])

    ax2 = sns.lineplot(x="target_vel", y="r2", hue=quantity, estimator=estimator, errorbar='sd', marker='o', data=df, legend='full');  #order=[10,9,8,7,6,5,4,3,2,1,0]

    leg=ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.xlabel('target velocity [deg/s]')
    plt.ylabel('R2')
    plt.grid()

    if quantity == 'sigma' and input_type == 'double_ring':
        leg.set_title(r'$u_0$')
    elif quantity == 'sigma' and input_type == 'random':
        leg.set_title(r'$\sigma$')
    else:
        leg.set_title(r'$g$')
        handles, labels = ax2.get_legend_handles_labels()
        new_labels = [label[:3] for label in labels]
        ax2.legend(handles, new_labels, title = r'$g$')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    plt.savefig(os.path.join(figpath, 'vary_{}_R2_vel.pdf'.format(quantity)), dpi=300, bbox_extra_artists=(leg,), bbox_inches='tight')
    plt.close()


def analyse_const_velocities(training_set, alpha, quantity, values, vels, params, fig_path):

    if training_set == 0:
        v_train = 360
    else:
        v_train = None

    if np.max(values) > 100:
        ylim = [-100, 100]
    else:
        ylim = None

    df = aggregate_linear_data(config.data_path, quantity, values, training_set, alpha, params, vels, config.n_sim, online=config.online)
    plot_delta_vel(df, quantity, params['input_structure'], fig_path, estimator='mean', ylim=ylim, v_train=v_train)
    plot_vel(df, quantity, params['input_structure'], fig_path, estimator='mean')
    plot_R2_vel(df, quantity, params['input_structure'], fig_path, estimator='mean')

    if training_set == 0:
        dfs = []
        for sigma in values:
            params['sigma'] = sigma
            network_params = utils.network_params_str(params)
            theory_path = os.path.join('theory', network_params)
            delta_psi = np.load(os.path.join(theory_path, 'delta_psi.npy'))
            theo_vels = np.load(os.path.join(theory_path, 'vels.npy'))

            theory_path = os.path.join(theory_path, 'v_train={}'.format(v_train))
            tangential_errors = np.load(os.path.join(theory_path, 'tangential_errors.npy'))
            radial_errors = np.load(os.path.join(theory_path, 'radial_errors.npy'))

            theo_vels_rad_ms = theo_vels * np.pi / (180. * 1000.) 
            #v_est = np.tan(np.arctan(params['tau'] * theo_vels_rad_ms) + tangential_errors[1]) / params['tau']
            #delta_vel = v_est - theo_vels_rad_ms
            #delta_vel *= 180. / np.pi * 1000.

            delta_vel = tangential_errors[1] / (params['tau'] * np.cos(delta_psi)**2)
            delta_vel *= 180. / np.pi * 1000.

            df_theo = pd.DataFrame(columns=['theo_sim', 'sigma', 'sim', 'target_vel', 'delta_vel'])
            df_theo['target_vel'] = theo_vels
            df_theo['delta_vel'] = delta_vel
            df_theo['sigma'] = sigma
            df_theo['theo_sim'] = 'theory'
            df_theo['sim'] = -1

            dfs.append(df_theo)

        # df_sim = df[df['sigma'] == params['sigma']].copy()
        # df_sim = df_sim[['sim', 'target_vel', 'delta_vel']].copy()
        # df_sim['theo_sim'] = 'simulation'

        # df_theo_sim = pd.concat([df_theo, df_sim], ignore_index=True)

        # plot_delta_vel(df_theo_sim, 'theo_sim', params['input_structure'], fig_path, estimator='mean', v_train=v_train)

        dfs = pd.concat(dfs, ignore_index=True)
        plot_delta_vel(dfs, 'sigma', params['input_structure'], fig_path, estimator='mean', fname='delta_vel_theory', ylim=ylim, v_train=v_train)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='plot performance')
    parser.add_argument('--set', type=int, required=False, default=0, help='set of training velocities')
    parser.add_argument('--alpha', type=float, required=False, default=1e-4, help='regularization parameter')
    parser.add_argument('--vary', type=str, required=False, default='sigma')
    parser.add_argument('--vals', type=int, nargs='+', required=False, default=[50,100,150,200,250], help='varied values')
    args = parser.parse_args()

    network_params = utils.network_params_str(config.params)

    characters_to_remove = None
    if args.vary == 'sigma':
        characters_to_remove = r'sigma={:.0f}_'.format(config.sigma)

    if characters_to_remove is not None and len(args.vals) > 1:
        network_params = re.sub(characters_to_remove, '', network_params)

    if config.online:
        fig_path = os.path.join('figs', network_params, 'closed_loop', 'online_alpha={:.0e}'.format(args.alpha))
    else:
        fig_path = os.path.join('figs', network_params, 'closed_loop', 'set={}_alpha={:.0e}'.format(args.set, args.alpha))
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    vels = np.hstack((-config.vels[::-1], config.vels))
    analyse_const_velocities(args.set, args.alpha, args.vary, args.vals, vels, config.params, fig_path)