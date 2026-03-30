import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib_config
import utils
import argparse
import config
from scipy import stats
import compare_readout_weights
import colorcet as cc


def plot_h(vels_theo, h, v_c, labels, fig_path):
    # plot h as a function of velocity
    fig, ax = plt.subplots(1,1, figsize=(2,1.5))
    fig.tight_layout()
    for i, (y, label) in enumerate(zip(h, labels)):
       ax.plot(vels_theo, y, label = label)
    ax.set_xlabel('Velocity [deg/s]')
    ax.set_ylabel(r'$h(v)$')
    ax.set_xlim(-v_c, v_c)
    ax.set_ylim(0, 1.1*np.max(h))
    ax.legend();
    plt.savefig(os.path.join(fig_path, 'h.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_lambda(sim_vels, lambdas_simulation, vels_theo, lambdas_theo, v_c, labels, fig_path, mode=1):
    # plot lambda as a function of velocity
    fig, ax = plt.subplots(1,1, figsize=(2,1.5))
    fig.tight_layout()
 
    for i, (lambda_sim, lambda_theo, label) in enumerate(zip(lambdas_simulation, lambdas_theo, labels)):
        ax.scatter(sim_vels, lambda_sim, marker='o', s=10)
        ax.plot(vels_theo, lambda_theo, label = label)

    ax.set_xlabel('velocity [deg/s]')
    ax.set_ylabel(r'$\lambda$')
    ax.set_ylim(0, 1.1*np.nanmax(lambdas_theo))
    ax.set_xlim(-v_c, v_c)
    ax.legend();
    plt.savefig(os.path.join(fig_path, 'lambda_mode={}.pdf'.format(mode)), dpi=300, bbox_inches='tight')
    plt.close()


def plot_phase(sim_vels, phase_sim, vels_theo, phase_theo, v_c, labels, fig_path, mode=1):
    # plot lambda as a function of velocity
    fig, ax = plt.subplots(1,1, figsize=(2,1.5))
    fig.tight_layout()

    for i, (xi_sim, xi_theo, label) in enumerate(zip(phase_sim, phase_theo, labels)):
        ax.scatter(sim_vels, xi_sim, marker='o', s=10)
        ax.plot(vels_theo, xi_theo)

    ax.set_xlabel('Velocity [deg/s]')
    ax.set_ylabel('Phase [rad]')
    ax.set_xlim(-v_c, v_c)

    if mode == 1:
        ax.set_ylim(-np.pi/2, np.pi/2)
        ax.set_yticks([-np.pi/2, 0, np.pi/2])
        ax.set_yticklabels([r'$-\frac{\pi}{2}$', r'0', r'$\frac{\pi}{2}$'])
    else:
        ax.set_ylim(-np.pi, np.pi)
        ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_yticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', r'0', r'$\frac{\pi}{2}$', r'$\pi$'])
    plt.savefig(os.path.join(fig_path, 'phase_mode={}.pdf'.format(mode)), dpi=300, bbox_inches='tight')
    plt.close()


def plot_lag(sim_vels, delta_psi, vels_theo, delta_psi_theo, v_c, fig_path):
    # plot lag as a function of velocity
    fig, ax = plt.subplots(1,1, figsize=(2,1.5))
    fig.tight_layout()

    ax.plot(vels_theo, delta_psi_theo, label = 'theory', zorder=1)
    ax.scatter(sim_vels, delta_psi, color='k', marker='o', s=matplotlib_config.MARKER_SIZE, label='simulation', zorder=2)

    ax.set_xlabel('velocity [deg/s]')
    ax.set_ylabel(r'$\Delta \psi$ [rad]')
    ax.set_xlim(-v_c, v_c)

    ax.set_ylim(-np.pi/2, np.pi/2)
    ax.set_yticks([-np.pi/2, 0, np.pi/2])
    ax.set_yticklabels([r'$-\frac{\pi}{2}$', r'0', r'$\frac{\pi}{2}$'])
    ax.legend();
    plt.savefig(os.path.join(fig_path, 'delta_psi.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_lag_continuum(psi_train, delta_psi_sim, delta_psi_theo, fig_path):
    # plot lag as a function of input location
    fig, ax = plt.subplots(1,1, figsize=(2,1.5))
    fig.tight_layout()

    ax.hlines(delta_psi_theo, xmin=-np.pi, xmax=np.pi, label = 'theory', zorder=1)
    ax.scatter(psi_train, delta_psi_sim, color='k', marker='o', s=1, label='simulation', zorder=2)

    ax.set_xlabel(r'$psi$ [rad]')
    ax.set_ylabel(r'$\Delta \psi$ [rad]')

    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels([r'-$\pi$', r'$-\frac{\pi}{2}$', r'0', r'$\frac{\pi}{2}$', r'$\pi$'])

    ax.set_ylim(-np.pi/2, np.pi/2)
    ax.set_yticks([-np.pi/2, 0, np.pi/2])
    ax.set_yticklabels([r'$-\frac{\pi}{2}$', r'0', r'$\frac{\pi}{2}$'])
    ax.legend();
    plt.savefig(os.path.join(fig_path, 'delta_psi.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_bump_shape(vels, s_driven, turn_ind, theta, vels_theo, lmbda_theo, xi_theo, delta_psi_theo, fig_path):

    n_vels = len(turn_ind)
    N = len(theta)

    cmap = plt.get_cmap('coolwarm')

    lmbda_minus, lmbda_plus = lmbda_theo[0], lmbda_theo[1]
    phase_minus, phase_plus = xi_theo[0], xi_theo[1]

    for i in range(n_vels):

        idx = np.where(vels_theo > vels[i])[0]

        if len(idx) > 0:

            idx = idx[0]
            lmbda = lmbda_plus[idx]
            xi = phase_plus[idx]
            dpsi = delta_psi_theo[idx]

            n_modes = len(lmbda)
            k = np.arange(n_modes)

            sp = 2 * lmbda[np.newaxis,1:] * np.cos(k[np.newaxis,1:] * (theta[:N//2,np.newaxis] + dpsi) - xi[np.newaxis,1:])
            sp = np.sum(sp, axis=-1) + lmbda[0].squeeze()

            lmbda = lmbda_minus[idx]
            xi = phase_minus[idx]

            sm = 2 * lmbda[np.newaxis,1:] * np.cos(k[np.newaxis,1:] * (theta[:N//2,np.newaxis] + dpsi) - xi[np.newaxis,1:])
            sm = np.sum(sm, axis=-1) + lmbda[0].squeeze()

            s_vel = s_driven[-turn_ind[i]-1,i]

            fig = plt.figure(figsize=(2.25,1.1))
            ax = fig.add_axes([0.2, 0.35, 0.55, 0.6]) 

            ax.plot(theta[:N//2], sp, 'k:', label="_nolegend_", zorder=2)
            ax.plot(theta[:N//2], sm, 'k:', label="_nolegend_", zorder=2)

            #ax.plot(vels_theo, delta_psi_theo, label = 'theory', zorder=1)
            ax.plot(theta[:N//2], s_vel[:N//2], lw=1, c=cmap(1.), label=' {:.0f}'.format(config.sigma), zorder=1)
            ax.plot(theta[N//2:], s_vel[N//2:], lw=1, c=cmap(0.), label='{:.0f}'.format(-config.sigma), zorder=1)

            ax.set_xlabel(r'$\theta$ [deg]')
            ax.set_ylabel(r'$s$ [a.u.]')
            ax.set_ylim([0, 1.1 * np.max(s_vel)])
            ax.set_xlim(-np.pi, np.pi)
            ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            ax.set_xticklabels([-180, -90, 0, 90, 180])

            direction = np.sign(vels[i])
            ax.vlines(0, ymin=0, ymax=1.1*s_vel.max(), color='dimgrey', linestyle='--', alpha=1., zorder=3)
            ax.arrow(x=0,y=2,dx=direction,dy=0,head_length=0.1,head_width=0.2, color='dimgrey', alpha=1., zorder=3)

            ax.legend(title=r'$u$', loc='upper left', bbox_to_anchor=(1.05, 1.0))

            plt.savefig(os.path.join(fig_path, 'bump_shape_vel={}.pdf'.format(vels[i])), dpi=300)
            plt.close()


def plot_readout_error(vels_theo, theo_radial_error, theo_angular_error, sim_vels, sim_radial_error, sim_angular_error, k, fname, fig_path, v_train=None):

    colors = ['tab:green', 'tab:purple']

    err_range = len(sim_vels) // 2 + np.arange(-15, 15)
    
    fig, ax = plt.subplots(1,1, figsize=(1.75,1.2))
    fig.tight_layout()
    if k == 0:
        ax.plot(vels_theo, theo_radial_error, c=colors[0], alpha=1.0, zorder=1)
        ax.scatter(sim_vels, sim_radial_error, marker='o', edgecolor='k', color=colors[0], linewidth=0.5, s=5)
        min_error = np.min(sim_radial_error[err_range])
        max_error = np.max(sim_radial_error[err_range])
        ax.set_ylim(min_error, max_error)
    else:
        ax.plot(vels_theo, theo_radial_error, c=colors[0], label='Radial error', alpha=1.0, zorder=1)
        ax.plot(vels_theo, theo_angular_error, c=colors[1], label='Tangential error', alpha=1.0, zorder=1)
        ax.scatter(sim_vels, sim_radial_error, marker='o', edgecolor='k', color=colors[0], linewidth=0.5, s=3)
        ax.scatter(sim_vels, sim_angular_error, marker='o', edgecolor='k', color=colors[1], linewidth=0.5, s=3)
        min_error = np.minimum(np.min(sim_radial_error[err_range]), np.min(sim_angular_error[err_range]))
        max_error = np.maximum(np.max(sim_radial_error[err_range]), np.max(sim_angular_error[err_range]))
        ax.set_ylim(min_error, max_error)
        ax.legend(bbox_to_anchor=(0.4, 1.1), loc='upper left');

    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.spines[['right', 'top']].set_visible(False)

    xmin = np.min(sim_vels[err_range])
    xmax = np.max(sim_vels[err_range])
    ax.axhline(0, linewidth=0.5, linestyle='--', color='k')
    if v_train is not None:
        ax.axvline(-v_train, linewidth=0.5, linestyle='--', color='k', zorder=-1)
        ax.axvline(v_train, linewidth=0.5, linestyle='--', color='k', zorder=-1)
    ax.set_xlabel(r'$v$ [deg/s]')
    ax.set_ylabel('Error')
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(360 * np.arange(-2, 3))
    plt.savefig(os.path.join(fig_path, fname), dpi=300, bbox_inches='tight')
    plt.close()


def theo_connectivity_doublering(theta, connectivity1, connectivity2):
    N = len(theta)
    W_rec_theo = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            delta_theta = theta[i] - theta[j]
            delta_theta = np.angle(np.exp(1j * delta_theta))
            theta_ind = np.argmin(np.abs(theta - delta_theta))
            if j < N // 2:
                W_rec_theo[i,j] = connectivity1[theta_ind]
            else:
                W_rec_theo[i,j] = connectivity2[theta_ind]

        #W_rec[:,i] = np.roll(W_rec[:,i], shift=config.N//2-i)

    return W_rec_theo


def plot_connectivity_scatter(W_rec, W_rec_theo, fig_path, fname='connectivity_comparison.png'):
    fig, ax = plt.subplots(figsize=(1.25,1.25))
    ax.scatter(W_rec.flatten(), W_rec_theo.flatten(), s=0.25, alpha=0.1, c='k')
    ax.set_yticks(ax.get_xticks())
    ax.set_ylim(ax.get_xlim())
    ax.set_aspect('equal', adjustable='box')
    plt.xlabel('Simulation weight')
    plt.ylabel('Theo. weight')
    plt.savefig(os.path.join(fig_path, fname), format='png', bbox_inches='tight', dpi=300)
    plt.close()


def plot_lambda_continuum(theo_weights, lmbda_theo, control_weights, lmbda_sim, fname, fig_path):
    # plot lmbda as a function of the control weight
    # compare the neuron-specific selectivity indices of the finite-size system
    # with the continuum solution for Fourier component of the ring associated with a specific control weight
    fig, ax = plt.subplots(1,1, figsize=(2,1.5))
    fig.tight_layout()
    ax.plot(theo_weights, lmbda_theo, label = 'theory', zorder=1)
    ax.scatter(control_weights, lmbda_sim, color='k', marker='o', s=0.5, label='simulation', zorder=2, alpha=0.5)
    ax.set_xlabel(r'$u$')
    ax.set_ylabel(r'$\lambda$')
    ax.legend();
    plt.savefig(os.path.join(fig_path, fname), dpi=300, bbox_inches='tight')
    plt.close()


def plot_xi_continuum(theo_weights, xi_theo, control_weights, xi_sim, fname, fig_path):
    fig, ax = plt.subplots(1,1, figsize=(2.0,1.5))
    fig.tight_layout()
    ax.plot(theo_weights, xi_theo, label = 'theory', zorder=1)
    ax.scatter(control_weights, xi_sim, color='k', marker='o', s=0.5, label='simulation', zorder=2, alpha=0.5)
    ax.set_xlabel(r'$u$')
    ax.set_ylabel(r'$\xi$')
    ax.legend();
    ax.set_ylim(-np.pi, np.pi)
    ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_yticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', r'0', r'$\frac{\pi}{2}$', r'$\pi$'])
    plt.savefig(os.path.join(fig_path, fname), dpi=300, bbox_inches='tight')
    plt.close()


def bump_shape_continuum_helper(fig, im, ax, w, sigma, nx, ny):
    ax.invert_yaxis()
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.set_ylabel(r'$u$')
    ax.set_xlabel(r'$\theta$ [deg]')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Activation [a.u.]')

    ny = len(w)
    w0_ratio = (w[-1] - 3 * sigma) / (w[-1] - w[0])
    step_ratio = sigma / (w[-1] - w[0])
    ax.set_yticks((w0_ratio + step_ratio * np.arange(7)) * ny)
    ax.set_yticklabels([r'$-3\sigma$', r'$-2\sigma$', r'$-\sigma$', r'$0$', r'$\sigma$', r'$2\sigma$', r'$3\sigma$'])
    ax.set_xticks([0, nx/4, nx/2, 3*nx/4, nx])
    ax.set_xticklabels([-180, -90, 0, 90, 180])
    ax.vlines(nx/2, ymin=0, ymax=ny, colors='k', linestyles='--')
    ax.set_ylim(w0_ratio * ny, (1 - w0_ratio) * ny)
    ax.set_xlim(0, nx)
    ax.arrow(nx/2, ny/2, nx/10, 0, head_width=0.5, head_length=10, overhang=0.5, color='darkgrey')

    return cbar


def get_bump_shape_continuum(w, theta, lmbda_theo, xi_theo, delta_psi, n_modes):

    k = np.arange(n_modes)
    s = np.zeros((len(w), len(theta)))

    for i, (lmbda, xi) in enumerate(zip(lmbda_theo, xi_theo)):
        si = 2 * lmbda[np.newaxis,1:] * np.cos(k[np.newaxis,1:] * (theta[:,np.newaxis] + delta_psi) - xi[np.newaxis,1:])
        s[i] = np.sum(si, axis=-1) + lmbda[0].squeeze()

    return s


def get_bump_derivative_continuum(w, theta, lmbda_theo, xi_theo, delta_psi, n_modes):

    k = np.arange(n_modes)
    s = np.zeros((len(w), len(theta)))

    for i, (lmbda, xi) in enumerate(zip(lmbda_theo, xi_theo)):
        si = -2 * lmbda[np.newaxis,1:] * k[np.newaxis,1:] * np.sin(k[np.newaxis,1:] * (theta[:,np.newaxis] + delta_psi) - xi[np.newaxis,1:])
        s[i] = np.sum(si, axis=-1)

    return s


def plot_theo_bump_shape_continuum(theta, w, lmbda_theo, xi_theo, delta_psi, sigma, n_modes, fname, fig_path, vmax=10):

    s = get_bump_shape_continuum(w, theta, lmbda_theo, xi_theo, delta_psi, n_modes)

    fig, ax = plt.subplots(1,1, figsize=(2,1.5))
    fig.subplots_adjust(left=0.02, bottom=0.25, right=0.95, top=0.75, wspace=0.05)
 
    ind = np.where(w > 3*sigma)[0][0]
    vmax = np.max(s[ind])
    im = ax.matshow(s, aspect=len(theta)/len(w), cmap=cc.cm.fire, interpolation='none', vmin=0, vmax=vmax)
    
    nx = len(theta)
    ny = len(w)
    bump_shape_continuum_helper(fig, im, ax, w, sigma, nx, ny)

    plt.savefig(os.path.join(fig_path, fname), dpi=300, bbox_inches='tight')
    plt.close()


def plot_theo_bump_derivative_continuum(ds, theta, w, sigma, fname, fig_path, vmax=10):

    fig, ax = plt.subplots(1,1, figsize=(2,1.5))
    fig.subplots_adjust(left=0.02, bottom=0.25, right=0.95, top=0.75, wspace=0.05)
 
    ind = np.where(w > 3*sigma)[0][0]
    vmax = np.max(np.abs(ds[ind]))
    im = ax.matshow(ds, aspect=len(theta)/len(w), cmap='berlin', interpolation='none', vmin=-vmax, vmax=vmax)
    
    nx = len(theta)
    ny = len(w)
    cbar = bump_shape_continuum_helper(fig, im, ax, w, sigma, nx, ny)
    cbar.set_label(r'$\frac{d s(\theta - \Delta \psi, u)}{d \theta}$ [a.u.]')

    plt.savefig(os.path.join(fig_path, fname), dpi=300, bbox_inches='tight')
    plt.close()


def plot_simulation_bump_shape_continuum(s_aligned, w_aligned, theta, w, lmbda_theo, xi_theo, delta_psi, sigma, n_modes, fname, fig_path, vmax=10):

    N = len(theta)
    theta_sim = np.tile(theta, N//2).flatten()

    w_sim = w_aligned.flatten()
    s_sim = s_aligned.flatten()

    dtheta = theta[1] - theta[0]
    theta_edges = np.hstack((theta, theta[-1] + dtheta)) - 0.5 * dtheta
    dw = w[1] - w[0]
    w_edges = np.hstack((w, w[-1] + dw)) - 0.5 * dw

    ret = stats.binned_statistic_2d(theta_sim, w_sim, s_sim, 'mean', bins=[theta_edges, w_edges])
    s_mean = ret.statistic.T

    s = get_bump_shape_continuum(w, theta, lmbda_theo, xi_theo, delta_psi, n_modes)

    fig, ax = plt.subplots(1,1, figsize=(2,1.5))
    fig.subplots_adjust(left=0.02, bottom=0.25, right=0.95, top=0.75, wspace=0.05)
 
    ind = np.where(w > 3*sigma)[0][0]
    vmax = np.max(s[ind])
    im = ax.matshow(s_mean, aspect=len(theta)/len(w), cmap=cc.cm.fire, interpolation='none', vmin=0, vmax=vmax)

    nx = len(theta)
    ny = len(w)
    bump_shape_continuum_helper(fig, im, ax, w, sigma, nx, ny)

    plt.savefig(os.path.join(fig_path, fname), dpi=300, bbox_inches='tight')
    plt.close()


def plot_example_bump_shape_continuum(s_sim, w_sim, theta, w, lmbda_theo, xi_theo, delta_psi, sigma, n_modes, fname, fig_path, vmax=10):

    s = get_bump_shape_continuum(w, theta, lmbda_theo, xi_theo, delta_psi, n_modes)

    fig, ax = plt.subplots(1,1, figsize=(2,1.5))
    fig.subplots_adjust(left=0.02, bottom=0.25, right=0.95, top=0.75, wspace=0.05)
 
    ind = np.where(w > 3*sigma)[0][0]
    vmax = np.max(s[ind])
    im = ax.matshow(np.nan * s, aspect=len(theta)/len(w), cmap=cc.cm.fire, interpolation='none', vmin=0, vmax=vmax)

    nx = len(theta)
    ny = len(w)
    bump_shape_continuum_helper(fig, im, ax, w, sigma, nx, ny)

    theta_scaled = nx * (0.5 + theta / (2 * np.pi))
    w_scaled = ny * (0.5 + w_sim / (w[-1] - w[0]))
    ax.scatter(theta_scaled, w_scaled, c=s_sim, cmap=cc.cm.fire, vmin=0, vmax=vmax)

    plt.savefig(os.path.join(fig_path, fname), dpi=300, bbox_inches='tight')
    plt.close()



def align_theo_simulation_connectivity_continuum(control_weights, W_rec, w, theo_connectivity):
    N = len(control_weights)
    W_rec_theo = np.zeros_like(W_rec)
    W_rec_aligned = W_rec.copy()

    for j in range(N):
        w_ind = np.argmin(np.abs(w - control_weights[j]))
        W_rec_theo[:,j] = theo_connectivity[w_ind]
        W_rec_aligned[:,j] = np.roll(W_rec_aligned[:,j], shift=N//2-j)

    return W_rec_aligned, W_rec_theo


def load_fourier(data_path):
    vels = np.load(os.path.join(data_path, 'vels.npy'))
    delta_psi = np.load(os.path.join(data_path, 'delta_psi.npy'))
    lmbda_minus = np.load(os.path.join(data_path, 'lmbda_minus.npy'))
    lmbda_plus = np.load(os.path.join(data_path, 'lmbda_plus.npy'))
    xi_minus = np.load(os.path.join(data_path, 'xi_minus.npy'))
    xi_plus = np.load(os.path.join(data_path, 'xi_plus.npy'))
    return vels, lmbda_plus, lmbda_minus, xi_plus, xi_minus, delta_psi


def load_rho_phi(data_path):
    rho = np.load(os.path.join(data_path, 'rho.npy'))
    phi = np.load(os.path.join(data_path, 'phi.npy'))
    return rho, phi


def load_readout_errors(data_path):
    radial_errors = np.load(os.path.join(data_path, 'radial_errors.npy'))
    tangential_errors = np.load(os.path.join(data_path, 'tangential_errors.npy'))
    return radial_errors, tangential_errors


def compare_readout_error(sim_path, theo_path, v_train, vels, vels_theo, fig_path_readout, example_sim=0):
    sim_ro_path = os.path.join(sim_path, 'sim={}'.format(example_sim), 'v_train={}'.format(v_train))
    radial_errors, tangential_errors = load_readout_errors(sim_ro_path)

    theo_ro_path = os.path.join(theo_path, 'v_train={}'.format(v_train))
    radial_errors_theo, tangential_errors_theo = load_readout_errors(theo_ro_path)

    n_modes = len(radial_errors_theo)

    for k in range(n_modes):
        fname = 'readout_error_v_train={}_mode={}.pdf'.format(v_train, k)
        plot_readout_error(vels_theo, radial_errors_theo[k], tangential_errors_theo[k], vels, radial_errors[k], tangential_errors[k], k, fname, fig_path_readout, v_train=v_train)


def comparison_doublering(sim_path, theo_path, s_driven, turn_ind, psi, theta, v_train, readout_weights, fb_weights, fig_path_fourier, fig_path_bump, fig_path_readout, fig_path_connectivity, example_sim=0):

    example_path = os.path.join(sim_path, 'sim={}'.format(example_sim))
    vels, lmbda_plus, lmbda_minus, xi_plus, xi_minus, delta_psi = load_fourier(example_path)
    vels_theo, lmbda_plus_theo, lmbda_minus_theo, xi_plus_theo, xi_minus_theo, delta_psi_theo = load_fourier(theo_path)

    labels = ['left ring', 'right ring']
    v_c = 900  # vels_theo[np.argmin(h[1])]
    n_modes = lmbda_plus.shape[1]

    for k in range(n_modes):
        lambda_simulation = [lmbda_minus[:,k], lmbda_plus[:,k]]
        lambda_theo = [lmbda_minus_theo[0,:,k], lmbda_plus_theo[0,:,k]]
        plot_lambda(vels, lambda_simulation, vels_theo, lambda_theo, v_c, labels, fig_path_fourier, mode=k)

        xi_simulation = [xi_minus[:,k], xi_plus[:,k]]
        xi_theo = [xi_minus_theo[0,:,k], xi_plus_theo[0,:,k]]
        plot_phase(vels, xi_simulation, vels_theo, xi_theo, v_c, labels, fig_path_fourier, mode=k)

    lambda_theo = [lmbda_minus_theo[0], lmbda_plus_theo[0]]
    xi_theo = [xi_minus_theo[0], xi_plus_theo[0]]

    plot_lag(vels, delta_psi, vels_theo, delta_psi_theo, v_c, fig_path_fourier)
    plot_bump_shape(vels, s_driven, turn_ind, theta, vels_theo, lambda_theo, xi_theo, delta_psi_theo, fig_path_bump)

    if readout_weights is not None:

        rho, phi = load_rho_phi(os.path.join(theo_path, 'v_train={}'.format(v_train)))
        #compare_readout_weights.plot_connectivity_scatter(config.J0 / config.N, readout_weights[0], fb_weights[0], theta, fig_path_readout)
        for m in range(n_modes):
            if m == 0:
                compare_readout_weights.compare_readout_weights_doublering(theta, readout_weights[0], rho[0,m], phi[0,m], v_train, m, fig_path_readout)
            else:
                compare_readout_weights.compare_readout_weights_doublering(theta, readout_weights[2*m-1:2*m+1], rho[0,m], phi[0,m], v_train, m, fig_path_readout)

        compare_readout_error(sim_path, theo_path, v_train, vels, vels_theo, fig_path_readout, example_sim=example_sim)

        W_rec = fb_weights.T @ readout_weights  # w/o J0
        compare_readout_weights.plot_weight_matrix(W_rec, fig_path_connectivity, fname='weight_matrix_sim.pdf')

        connectivity_minus = np.load(os.path.join(theo_path, 'v_train={}'.format(v_train), 'connectivity_minus.npy'))
        connectivity_plus = np.load(os.path.join(theo_path, 'v_train={}'.format(v_train), 'connectivity_plus.npy'))

        W_rec_theo = theo_connectivity_doublering(theta, connectivity_minus, connectivity_plus)

        compare_readout_weights.plot_weight_matrix(W_rec_theo, fig_path_connectivity, fname='weight_matrix_theo.pdf')

        plot_connectivity_scatter(W_rec, W_rec_theo, fig_path_connectivity)


def comparison_continuum(sim_path, theo_path, vels, s_driven, turn_ind, psi, theta, v_train, control_weights, readout_weights, fb_weights, fig_path_fourier, fig_path_bump, fig_path_readout, fig_path_connectivity, example_sim=0):

    example_path = os.path.join(sim_path, 'sim={}'.format(example_sim), 'v_train={}'.format(v_train))
    delta_psi = np.load(os.path.join(example_path, 'delta_psi.npy'))
    lmbda = np.load(os.path.join(example_path, 'lmbda.npy'))
    xi = np.load(os.path.join(example_path, 'xi.npy'))

    vels_theo, lmbda_plus_theo, lmbda_minus_theo, xi_plus_theo, xi_minus_theo, delta_psi_theo = load_fourier(theo_path)
    lmbda_theo = np.vstack((lmbda_minus_theo[::-1], lmbda_plus_theo))
    xi_theo = np.vstack((xi_minus_theo[::-1], xi_plus_theo))
    w = np.load(os.path.join(theo_path, 'w_values.npy'))

    vel_ind = np.where(vels_theo > v_train)[0][0]
    train_id = np.where(vels==v_train)[0][0]
    s_train = s_driven[-turn_ind[train_id]-1:-1, train_id]
    psi_train = psi[-turn_ind[train_id]:, train_id]
    psi_train = np.angle(np.exp(1j * psi_train))
    plot_lag_continuum(psi_train, delta_psi, delta_psi_theo[vel_ind], fig_path_fourier)

    for k in range(config.n_modes):
        fname = 'lambda_mode={}.pdf'.format(k)
        plot_lambda_continuum(w, lmbda_theo[:,vel_ind,k].squeeze(), control_weights, lmbda[:,k], fname, fig_path_fourier)

        fname = 'phase_mode={}.pdf'.format(k)
        plot_xi_continuum(w, xi_theo[:,vel_ind,k].squeeze(), control_weights, xi[:,k], fname, fig_path_fourier)

    fname = 'activity_pattern.pdf'
    plot_theo_bump_shape_continuum(theta, w, lmbda_theo[:,vel_ind,:], xi_theo[:,vel_ind,:], delta_psi_theo[vel_ind], config.sigma, config.n_modes, fname, fig_path_bump)

    ds = get_bump_derivative_continuum(w, theta, lmbda_theo[:,vel_ind,:], xi_theo[:,vel_ind,:], delta_psi_theo[vel_ind], config.n_modes)
    np.save(os.path.join(theo_path, 'ds.npy'), ds)

    fname = 'derivative_activity_pattern.pdf'
    plot_theo_bump_derivative_continuum(ds, theta, w, config.sigma, fname, fig_path_bump)

    delta = int((360 / abs(v_train)) * ((1000 / config.dt) / (config.N // 2)))
    s_aligned = s_train[::delta,:].copy()
    control_weights_aligned = np.zeros_like(s_aligned)
    for k in np.arange(config.N//2):
        s_aligned[k] = np.roll(s_aligned[k], -2*k)
        control_weights_aligned[k] = np.roll(control_weights, -2*k)

    fname = 'activity_pattern_simulation.pdf'
    plot_simulation_bump_shape_continuum(s_aligned, control_weights_aligned, theta, w, lmbda_theo[:,vel_ind,:], xi_theo[:,vel_ind,:], delta_psi_theo[vel_ind], config.sigma, config.n_modes, fname, fig_path_bump)

    loc_id = np.argmin(np.abs(psi_train))
    s_example = s_train[loc_id]

    fname = 'activity_pattern_example.pdf'
    plot_example_bump_shape_continuum(s_example, control_weights, theta, w, lmbda_theo[:,vel_ind,:], xi_theo[:,vel_ind,:], delta_psi_theo[vel_ind], config.sigma, config.n_modes, fname, fig_path_bump)

    rho, phi = load_rho_phi(os.path.join(theo_path, 'v_train={}'.format(v_train)))


    if readout_weights is not None:

        for m in range(config.n_modes):

            if m == 0:
                readout_numerical = readout_weights.squeeze()[[0]]
            else:
                readout_numerical = readout_weights.squeeze()[[2*m-1,2*m]]

            rho_simulation, phi_simulation = compare_readout_weights.comp_sim_readout_params_continuum(theta, readout_numerical, m)

            fname = 'phi_vel={:.0f}_mode={:.0f}.pdf'.format(v_train, m)
            compare_readout_weights.compare_phi_continuum(w, phi[:,m], control_weights, phi_simulation, fname, fig_path_readout)

            fname = 'rho_vel={:.0f}_mode={:.0f}.pdf'.format(v_train, m)
            compare_readout_weights.compare_rho_continuum(w, rho[:,m], control_weights, rho_simulation, fname, fig_path_readout)

            if m == 0 :
                fname = 'readout_weights_vel={:.0f}_mode={:.0f}.pdf'.format(v_train, m)
            else:
                fname = 'cos_readout_weights_vel={:.0f}_mode={:.0f}.pdf'.format(v_train, m)
            compare_readout_weights.compare_readout_weights_random(w, m, rho[:,m], phi[:,m], theta, control_weights, readout_numerical[0], fname, fig_path_readout)

        compare_readout_error(sim_path, theo_path, v_train, vels, vels_theo, fig_path_readout, example_sim=example_sim)

        W_rec = fb_weights.T @ readout_weights  # w/o J0
        compare_readout_weights.plot_weight_matrix(W_rec, fig_path_connectivity, double_ring=False)

        theo_connectivity = np.load(os.path.join(theo_path, 'v_train={}'.format(v_train), 'connectivity.npy'))
        compare_readout_weights.plot_theo_connectivity_continuum(w, theta, theo_connectivity, config.sigma, fig_path_connectivity)
        compare_readout_weights.plot_simulation_connectivity_continuum(w, theta, config.sigma, control_weights, W_rec, fig_path_connectivity, connectivity=theo_connectivity, fname='connectivity_simulation.pdf')

        W_rec, W_rec_theo = align_theo_simulation_connectivity_continuum(control_weights, W_rec, w, theo_connectivity)
        plot_connectivity_scatter(W_rec, W_rec_theo, fig_path_connectivity)


def get_fig_paths(network_params):
    fig_path_fourier = os.path.join('figs', network_params, 'fourier_modes')
    if not os.path.exists(fig_path_fourier):
        os.makedirs(fig_path_fourier)

    fig_path_bump = os.path.join('figs', network_params, 'bump_shape')
    if not os.path.exists(fig_path_bump):
        os.makedirs(fig_path_bump)

    fig_path_readout = os.path.join('figs', network_params, 'readout')
    if not os.path.exists(fig_path_readout):
        os.makedirs(fig_path_readout)

    fig_path_connectivity = os.path.join('figs', network_params, 'connectivity')
    if not os.path.exists(fig_path_connectivity):
        os.makedirs(fig_path_connectivity)

    return fig_path_fourier, fig_path_bump, fig_path_readout, fig_path_connectivity



def main():
    parser = argparse.ArgumentParser(description='Compute macroscopic quantities.')
    parser.add_argument('--sim', type=int, required=False, default=0, help='example simulation number')
    parser.add_argument('--alpha', type=float, required=False, default=1e-4, help='regularization parameter')
    parser.add_argument('--vel', type=int, required=False, default=360, help='training velocity')
    args = parser.parse_args()

    network_params = utils.network_params_str(config.params)
    fig_path_fourier, fig_path_bump, fig_path_readout, fig_path_connectivity = get_fig_paths(network_params)

    sim_path = os.path.join('simulation', network_params)
    theo_path = os.path.join('theory', network_params)

    data_path = utils.get_data_path(config.data_path, config.params)
    data_path = os.path.join(data_path, 'sim={}'.format(args.sim))

    try:
        s_driven = np.load(os.path.join(data_path, 's_driven.npy'), mmap_mode='r')
        turn_ind = np.load(os.path.join(data_path, 'turn_ind.npy'))
        psi = np.load(os.path.join(data_path, 'psi.npy'))
        vels = np.load(os.path.join(data_path, 'vels.npy'))

        theta = np.load(os.path.join(data_path, 'theta.npy'))
        control_weights = np.load(os.path.join(data_path, 'control_weights.npy')).squeeze()

        train_set = utils.get_train_set_index_single_vel(args.vel)

        fit_path = os.path.join(data_path, 'set={}_alpha={:.0e}'.format(train_set, args.alpha))

        with open(os.path.join(fit_path, 'fit_successful.txt')) as f:
            fit_successful = int(f.readlines()[0])

        if fit_successful:
            readout_weights = np.load(os.path.join(fit_path, 'readout_weights.npy'))
            fb_weights = np.load(os.path.join(data_path, 'feedback_weights.npy'))
        else:
            readout_weights = None
            fb_weights = None

    except FileNotFoundError as e:
        raise FileNotFoundError(
            'Simulation data does not exist: {}'.format(data_path)
        ) from e

    if config.input_structure == 'double_ring':
        comparison_doublering(sim_path, theo_path, s_driven, turn_ind, psi, theta, args.vel,readout_weights, fb_weights, fig_path_fourier, fig_path_bump, fig_path_readout, fig_path_connectivity, example_sim=args.sim)
    else:
        comparison_continuum(sim_path, theo_path, vels, s_driven, turn_ind, psi, theta, args.vel, control_weights, readout_weights, fb_weights, fig_path_fourier, fig_path_bump, fig_path_readout, fig_path_connectivity, example_sim=args.sim)


if __name__ == "__main__":
    main()