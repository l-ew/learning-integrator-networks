import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import matplotlib_config
import visualization
import config
import utils


def get_bump_shapes(s_driven, z, turn_ind):
    n_vels = z.shape[1]
    N =s_driven.shape[2]
    bump_shapes = np.zeros((n_vels,N))
    for m in range(n_vels):
        k = turn_ind[m]
        z_v = z[-k+1:,m,1] + 1j * z[-k+1:,m,2]
        psi_v = np.angle(z_v)
        loc_id = np.argmin(np.abs(psi_v))
        bump_shapes[m] = s_driven[loc_id,m]
    return bump_shapes


def plot_bump_shape(theta, bump_shape, fig_path, w=None, display_loc=True, ymax=None, fname='bump_shape.pdf'):
    fig = plt.figure(figsize=(2.25,1.1))
    ax = fig.add_axes([0.2, 0.35, 0.55, 0.6]) 

    if w is not None:
        vmax = np.maximum(np.max(w), -np.min(w))
        im = ax.scatter(theta, bump_shape, marker='.', edgecolors='none', c=w, s=5.0, alpha=1.0, vmin=-vmax, vmax=vmax, cmap='coolwarm', zorder=-1)
    else:
        im = ax.scatter(theta, bump_shape, marker='.', edgecolors='none', s=5.0, alpha=1.0, zorder=-1)


    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], labels=[-180, -90, 0, 90, 180]);

    #bump_loc = np.angle(np.sum(np.exp(1j * theta) * bump_shape))
    #ax.vlines(bump_loc, ymin=0, ymax=1.1*bump_shape.max(), color='k', linestyle='--', alpha=0.5)

    ax.set(xlabel=r'$\theta$ [deg]', ylabel='')
    ax.set_ylabel('Activation [a.u.]')
    if ymax is None:
        ax.set_ylim(0, 1.1*bump_shape.max())
    else:
        ax.set_ylim(0, ymax)

    if display_loc:
        ax.vlines(0, ymin=0, ymax=1.1*bump_shape.max(), color='dimgrey', linestyle='--', alpha=1.)
        ax.arrow(x=0,y=2,dx=1,dy=0,head_length=0.1,head_width=0.2, color='dimgrey', alpha=1.)

    if w is not None:
        cax = fig.add_axes([0.775, 0.4, 0.015, 0.475])
        cb=plt.colorbar(im, cax=cax)
        cb.set_label(r'$u$ [a.u.]')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(os.path.join(fig_path, fname), dpi=300)
    plt.close()


def plot_bump_shapes(theta, bump_shapes, vels, fig_path, w=None, color_w=True, display_loc=False, n_rows=3, fname='bump_shapes.pdf'):
    n_vels, N = bump_shapes.shape
    fig, axs = plt.subplots(n_rows, 2, figsize=(2,5))
    fig.tight_layout()

    if w is not None:
        vmax = np.maximum(np.max(w), -np.min(w))

    for m in range(n_rows):
        if color_w:
            axs[m,0].scatter(theta, bump_shapes[n_vels//2+m, :], c=w, alpha=0.5, s=5, vmin=-vmax, vmax=vmax, cmap='coolwarm')
        else:
            axs[m,0].plot(theta, bump_shapes[n_vels//2+m, :])
        axs[m,0].set_xlim(-np.pi, np.pi)
        axs[m,0].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], labels=[-180, -90, 0, 90, 180]);
        axs[m,0].set_title('v = {:.0f} [deg/s]'.format(vels[n_vels//2+m]))
        if display_loc:
            bump_loc = np.angle(np.sum(np.exp(1j * theta) * bump_shapes[n_vels//2+m, :N//2]))
            axs[m,0].vlines(bump_loc, ymin=0, ymax=1.1*bump_shapes[n_vels//2].max(), color='k', linestyle='--', alpha=0.5)
    
        
    for m in range(n_rows):
        if color_w:
            axs[m,1].scatter(theta, bump_shapes[n_vels//2-m-1, :], c=w, alpha=0.5, s=5,  vmin=-vmax, vmax=vmax, cmap='coolwarm')
        else:
            axs[m,1].plot(theta, bump_shapes[n_vels//2-m-1, :])
        axs[m,1].set_xlim(-np.pi, np.pi)
        axs[m,1].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], labels=[-180, -90, 0, 90, 180]);
        axs[m,1].set_title('v = {:.0f} [deg/s]'.format(vels[n_vels//2-m-1]))
        if display_loc:
            bump_loc = np.angle(np.sum(np.exp(1j * theta) * bump_shapes[n_vels//2-m-1, :N//2]))
            axs[m,1].vlines(bump_loc, ymin=0, ymax=1.1*bump_shapes[n_vels//2].max(), color='k', linestyle='--', alpha=0.5)


    for ax in axs.flat:
        ax.set(xlabel=r'$\theta$ [deg]', ylabel='')
        ax.set_ylabel('Activation [a.u.]')
        ax.vlines(0, ymin=0, ymax=1.1*bump_shapes[n_vels//2].max(), color='k', linestyle='--', alpha=0.5)
        ax.set_ylim(0, 1.1*bump_shapes[n_vels//2].max())

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.savefig(os.path.join(fig_path, fname), dpi=300, bbox_inches='tight')
    plt.close()

    if color_w:
        a = np.array([[0,1]])
        plt.figure(figsize=(3, 0.6))
        img = plt.imshow(a, cmap='coolwarm', vmin=-vmax, vmax=vmax)
        plt.gca().set_visible(False)
        cax = plt.axes([0.1, 0.05, 0.8, 0.35])
        cbar = plt.colorbar(cax=cax, orientation='horizontal')
        cbar.set_label('velocity input weight');
        plt.savefig(os.path.join(fig_path,"colorbar.pdf"), bbox_inches='tight')
        plt.close()



def plot_bump_shapes_doublering(theta, bump_shapes, vels, fig_path, display_loc=False):
    n_vels, N = bump_shapes.shape
    n_rows = 3
    fig, axs = plt.subplots(n_rows, 2, figsize=(8,8))
    fig.tight_layout()

    for m in range(n_rows):
        axs[m,0].plot(theta, bump_shapes[n_vels//2+m, :N//2])
        axs[m,0].plot(theta, bump_shapes[n_vels//2+m, N//2:])
        axs[m,0].set_xlim(-np.pi, np.pi)
        axs[m,0].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], labels=[-180, -90, 0, 90, 180]);
        axs[m,0].set_title('v = {:.0f} [deg/s]'.format(vels[n_vels//2+m]))
        if display_loc:
            bump_loc = np.angle(np.sum(np.exp(1j * theta) * bump_shapes[n_vels//2+m, :N//2]))
            axs[m,0].vlines(bump_loc, ymin=0, ymax=1.1*bump_shapes[n_vels//2].max(), color='k', linestyle='--', alpha=0.5)
    
        
    for m in range(n_rows):
        axs[m,1].plot(theta, bump_shapes[n_vels//2-m-1, :N//2])
        axs[m,1].plot(theta, bump_shapes[n_vels//2-m-1, N//2:])
        axs[m,1].set_xlim(-np.pi, np.pi)
        axs[m,1].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], labels=[-180, -90, 0, 90, 180]);
        axs[m,1].set_title('v = {:.0f} [deg/s]'.format(vels[n_vels//2-m-1]))
        if display_loc:
            bump_loc = np.angle(np.sum(np.exp(1j * theta) * bump_shapes[n_vels//2-m-1, :N//2]))
            axs[m,1].vlines(bump_loc, ymin=0, ymax=1.1*bump_shapes[n_vels//2].max(), color='k', linestyle='--', alpha=0.5)


    for ax in axs.flat:
        ax.set(xlabel=r'$\theta$ [deg]', ylabel='')
        ax.set_ylabel('Activation [a.u.]')
        ax.vlines(0, ymin=0, ymax=1.1*bump_shapes[n_vels//2].max(), color='r', linestyle='--', alpha=0.5)
        ax.set_ylim(0, 1.1*bump_shapes[n_vels//2].max())

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.savefig(os.path.join(fig_path, 'bump_shapes.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_bump_shape_doublering(theta, bump_shape, fig_path, w=None, fname='bump_shape.pdf'):
    N = len(bump_shape)
    fig = plt.figure(figsize=(2.25,1.1))
    ax = fig.add_axes([0.2, 0.35, 0.55, 0.6]) 

    cmap = plt.get_cmap('coolwarm')

    ax.plot(theta[:N//2], bump_shape[:N//2], c=cmap(1.), label=' {:.0f}'.format(np.max(w)))
    ax.plot(theta[N//2:], bump_shape[N//2:], c=cmap(0.), label='{:.0f}'.format(np.min(w)))
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], labels=[-180, -90, 0, 90, 180]);

    ax.set(xlabel=r'$\theta$ [deg]', ylabel='')
    ax.set_ylabel('Activation [a.u.]')
    ax.set_ylim(0, 1.1*bump_shape.max())

    ax.vlines(0, ymin=0, ymax=1.1*bump_shape.max(), color='dimgrey', linestyle='--', alpha=1., zorder=3)
    ax.arrow(x=0,y=2,dx=1,dy=0,head_length=0.1,head_width=0.2, color='dimgrey', alpha=1., zorder=3)

    ax.legend(title=r'$u$ [a.u.]', loc='upper left', bbox_to_anchor=(1.05, 1.0))

    plt.savefig(os.path.join(fig_path, fname), dpi=300)
    plt.close()



def main(training_set, alpha, sim):

    network_params = utils.network_params_str(config.params)
    main_path = os.path.join(config.data_path, network_params)

    vels = np.hstack((-config.vels[::-1], config.vels))

    data_path = os.path.join(main_path, 'sim={}'.format(sim))
    t_driven = np.load(os.path.join(data_path, 't.npy'))
    s_driven = np.load(os.path.join(data_path, 's_driven.npy'))
    turn_ind = np.load(os.path.join(data_path, 'turn_ind.npy'))
    theta = np.load(os.path.join(data_path, 'theta.npy'))
    control_weights = np.load(os.path.join(data_path, 'control_weights.npy'))
    z = np.load(os.path.join(data_path, 'z.npy'))

    fig_path = os.path.join('figs', network_params, 'driven_network', 'set={}_alpha={}'.format(training_set, alpha))
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    bump_shapes = get_bump_shapes(s_driven, z, turn_ind)
    mask = np.abs(vels) > 350  # only use large velocitites
    vels = vels[mask]
    bump_shapes = bump_shapes[mask]
    s_driven = s_driven[:,mask]
    turn_ind = turn_ind[mask]
    n_vels = np.sum(mask)  #len(vels)

    if config.input_structure == 'double_ring':
        plot_bump_shapes_doublering(theta[:config.N//2], bump_shapes, vels, fig_path, display_loc=False)

        plot_bump_shape_doublering(theta, bump_shapes[n_vels//2, :], fig_path, w=control_weights)

        k = int(3000 / config.dt)
        sv = s_driven[:k,n_vels//2]  #-turn_ind[n_vels//2]:
        t = t_driven[:k]

        visualization.plot_act_over_time(t, sv, fig_path, xticks=[0, 1,2,3], double_ring=True)
    else:
        if not config.theta_grid:
            ordering = np.argsort(theta)
            theta = theta[ordering]
            s_driven = s_driven[:,:,ordering]
            bump_shapes = bump_shapes[:,ordering]
            control_weights = control_weights[ordering]

        plot_bump_shape(theta, bump_shapes[n_vels//2, :], fig_path, w=control_weights, display_loc=True)
        #plot_bump_shapes(theta, bump_shapes, vels, fig_path, w=control_weights, color_w=True, display_loc=False)


        k = int(3000 / config.dt)
        sv = s_driven[:k,n_vels//2]  #-turn_ind[n_vels//2]:
        t = t_driven[:k]
        visualization.plot_act_over_time(t, sv, fig_path, xticks=[0, 1,2,3])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulate closed loop system.')
    parser.add_argument('--set', type=int, required=False, default=0, help='set of training velocities')
    parser.add_argument('--alpha', type=float, required=False, default=1e-4, help='regularization parameter')
    parser.add_argument('--sim', type=int, required=False, default=0, help='simulation number')
    args = parser.parse_args()
    main(args.set, args.alpha, args.sim)
