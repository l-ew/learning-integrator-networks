import numpy as np
import os
import colorcet as cc
import matplotlib.pyplot as plt
import matplotlib_config
from scipy.optimize import minimize
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy import stats
from scipy.stats import norm
import config


def readout_fun(rho, phi, theta, N):
    y = np.ones(N, dtype='complex')
    y[:N//2] = rho * np.exp(1j * theta[:N//2]) * np.exp(-1j * phi) / (N//2)
    y[N//2:] = rho * np.exp(1j * theta[N//2:]) * np.exp(1j * phi) / (N//2)
    return y


def loss_fun(x, N, theta, readout_weight):
    y = readout_fun(x[0], x[1], theta, N)
    loss = np.sum((np.real(y) - readout_weight[0])**2) + np.sum((np.imag(y) - readout_weight[1])**2)
    return loss


def fit_readout_weights_doublering(theta, readout_weight):
    x0 = [1, 0]
    N = len(theta)
    res = minimize(lambda x: loss_fun(x, N, theta, readout_weight), x0, method='Nelder-Mead', tol=1e-6)
    rho = res.x[0]
    phi = res.x[1]
    return rho, phi


def comp_sim_readout_params_doublering(vels, theta, readout_weights):
    rhos = np.zeros(len(vels))
    phis = np.zeros(len(vels))

    for i in range(len(vels)):
        rho, phi = fit_readout_weights_doublering(theta, readout_weights[i])
        rhos[i] = rho
        phis[i] = phi
        
    return rhos, phis


def plot_readout_params_across_velocities_doublering(vels, phi_theo, phi_sim, rho_theo, rho_sim, alpha, fig_path):

    fig, ax = plt.subplots(1,1, figsize=(5,3))
    fig.tight_layout()

    ax.plot(vels, phi_theo * 180 / np.pi, '--.', label='analytical')
    ax.plot(vels, phi_sim * 180 / np.pi, '--.', label='simulation')
    ax.set_xticks(vels[::2])
    ax.set_xlabel('velocity [deg / s]')
    ax.set_ylabel(r'$\phi$ [deg]');

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.1, 0.95))

    plt.savefig(os.path.join(fig_path, 'phi_alpha={:.0f}_vary_velocity.pdf'.format(alpha)), dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(1,1, figsize=(5,3))
    fig.tight_layout()

    ax.plot(vels, rho_theo, '--.', label='analytical')
    ax.plot(vels, rho_sim, '--.', label='simulation')
    ax.set_xticks(vels[::2])
    ax.set_xlabel('velocity [deg / s]')
    ax.set_ylabel(r'$\rho$');

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.1, 0.95))

    plt.savefig(os.path.join(fig_path, 'rho_alpha={:.0f}_vary_velocity.pdf'.format(alpha)), dpi=300, bbox_inches='tight')
    plt.close()

def plot_readout_params_across_alpha_doublering(alphas, phi_theo, phi_sim, rho_theo, rho_sim, vel, fig_path):

    fig, ax = plt.subplots(1,1, figsize=(5,3))
    fig.tight_layout()

    ax.plot(alphas, phi_theo * 180 / np.pi, '--.', label='analytical')
    ax.plot(alphas, phi_sim * 180 / np.pi, '--.', label='simulation')
    ax.set_xticks(alphas[::2])
    ax.set_xscale('log')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\phi$ [deg]');

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.1, 0.95))

    plt.savefig(os.path.join(fig_path, 'phi_vel={:.0f}_vary_alpha.pdf'.format(vel)), dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(1,1, figsize=(5,3))
    fig.tight_layout()

    ax.plot(alphas, rho_theo, '--.', label='analytical')
    ax.plot(alphas, rho_sim, '--.', label='simulation')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\rho$');

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.1, 0.95))

    plt.savefig(os.path.join(fig_path, 'rho_vel={:.0f}_vary_alpha.pdf'.format(vel)), dpi=300, bbox_inches='tight')
    plt.close()


def plot_weight_matrix(W, fig_path, fname='weight_matrix.pdf', double_ring=True):

    N = W.shape[1]

    fig, ax = plt.subplots(figsize=(1.5,1.5))

    #plt.axis('off')
    vmax = np.maximum(np.max(W), -np.min(W))
    im=ax.imshow(W, interpolation=None, cmap='coolwarm', aspect=1, vmin=-vmax, vmax=vmax)
    if double_ring:
        plt.plot([0, N], [N/2, N/2], 'k--', linewidth=0.75)
        plt.plot([N/2, N/2], [0, N], 'k--', linewidth=0.75)
    ax.set_xlabel('Presynaptic cell #')
    ax.set_ylabel('Postsynaptic cell #')
    ax.set_xlim(0,N)
    ax.set_ylim(0,N)
    cb=plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04) #ticks=[J0-0.05, J0, J0+0.05])
    cb.set_label('Recurrent weight')

    plt.savefig(os.path.join(fig_path, fname), format='pdf', bbox_inches='tight', dpi=300)
    plt.close()


def plot_connectivity_scatter(J0, readout_weights, fb_weights, theta, fig_path, fname='weight_function.png'):

    tot_weights = J0 + fb_weights.T @ readout_weights

    N = fb_weights.shape[1]

    fig, ax = plt.subplots(figsize=(4,3))

    for k in range(N):
        ax.scatter(theta.squeeze(), np.roll(tot_weights[k], N//2-k).squeeze(), s=1, alpha=0.5, c='k')
    #plt.xlabel(r'$\theta - \theta^\prime$')
    #plt.ylabel(r'$W_{REC}(\theta - \theta^\prime)$')
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', r'0', r'$\frac{\pi}{2}$', r'$\pi$'])

    plt.savefig(os.path.join(fig_path, fname), format='png', bbox_inches='tight', dpi=300)
    plt.close()


def compare_readout_weights_random(w, k, rho, phi, theta, w_simulation, readout_weights, fname, fig_path):

    fig, ax = plt.subplots(figsize=(5,3))
    plt.subplots_adjust(right=0.85)

    vmax = np.maximum(np.max(w_simulation), -np.min(w_simulation))

    normalize = Normalize(vmin=-vmax, vmax=vmax)
    scalarMap = ScalarMappable(norm=normalize, cmap="coolwarm")

    N = len(theta)

    w_samples = [-100, -50, -25, 0, 25, 50, 100]

    if config.input_structure == 'multi_ring':
        M = int(np.sqrt(config.N))
        q = np.linspace(0.5/M, 1.0-0.5/M, M)
        w_samples = norm.ppf(q, 0, scale=config.sigma)[2::5]

    for w1 in w_samples:
        w1_ind = np.argmin(np.abs(w1-w))
        weights_theo = rho[w1_ind] * np.exp(1j * (k * theta + phi[w1_ind])) / (N//2)
        y = np.real(weights_theo)
        ax.plot(theta, y, zorder=1, linewidth=1, c=scalarMap.to_rgba(w1), alpha=1)

    im=ax.scatter(theta, readout_weights, s=2, alpha=1, c=w_simulation, zorder=2, cmap=scalarMap.cmap, vmin=-vmax, vmax=vmax)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', r'0', r'$\frac{\pi}{2}$', r'$\pi$'])
    ax.set_xlabel(r'$\theta$ [rad]')
    ax.set_ylabel(r'$p(\theta)$')

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    cb=plt.colorbar(im, fraction=0.02, pad=0.04)
    cb.set_label(r'$w$')

    plt.savefig(os.path.join(fig_path, fname), dpi=300, bbox_inches='tight')
    plt.close()


def compare_phi_continuum(w_theo, phi_theo, w_simulation, phi_simulation, fname, fig_path):
    fig, ax = plt.subplots(1,1, figsize=(6,3))
    ax.grid('on', alpha=0.5)
    ax.plot(w_theo, phi_theo * 180 / np.pi, c='k', label='theory')
    ax.scatter(w_simulation, phi_simulation * 180 / np.pi, alpha=0.5, s=1, label='simulation')
    ax.set_xlim(np.min(w_theo), np.max(w_theo))
    ax.set_ylim(-90, 90)
    ax.set_yticks([-90, -45, 0, 45, 90])
    ax.set_xlabel(r'$w$')
    ax.set_ylabel(r'$\phi$')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(fig_path, fname), dpi=300, bbox_inches='tight')
    plt.close()


def compare_rho_continuum(w_theo, rho_theo, w_simulation, rho_simulation, fname, fig_path):
    fig, ax = plt.subplots(1,1, figsize=(6,3))
    ax.grid('on', alpha=0.5)
    ax.plot(w_theo, rho_theo, c='k', label='theory')
    ax.scatter(w_simulation, rho_simulation, alpha=0.5, s=1, label='simulation')
    ax.set_xlim(np.min(w_theo), np.max(w_theo))
    ax.set_xlabel(r'$w$')
    ax.set_ylabel(r'$\rho$')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(fig_path, fname), dpi=300, bbox_inches='tight')
    plt.close()

def comp_sim_readout_params_continuum(theta, readout_weights, k):
    if k > 0:
        z = readout_weights[0,:] + 1j * readout_weights[1,:]
    else:
        z = readout_weights[0,:]
    N = len(theta)
    z = N // 2 * z * np.exp(-1j * k * theta)
    rho_simulation = np.abs(z)
    phi_simulation = np.angle(z)
    return rho_simulation, phi_simulation



def plot_theo_connectivity_continuum(w, theta, connectivity, sigma, fig_path, fname='connectivity.pdf'):

    ny, nx = connectivity.shape

    fig, ax = plt.subplots(1,1, figsize=(2,1.5))
    fig.subplots_adjust(left=0.02, bottom=0.25, right=0.95, top=0.75, wspace=0.05)

    w_thr = 3 * sigma

    ind = np.where(w > w_thr)[0][0]
    vmax = np.max(connectivity[ind])
    #vmin = np.min(connectivity[ind])
    im = ax.matshow(connectivity, aspect=len(theta)/len(w), cmap='coolwarm', interpolation='none', vmin=-vmax, vmax=vmax) 
    ax.invert_yaxis()
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.set_ylabel(r'$u_{\mathrm{pre}}$')
    ax.set_xlabel(r'$\Delta \theta = \theta_{\mathrm{post}} - \theta_{\mathrm{pre}}$ [deg]')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Recurrent weight')

    ny = len(w)
    w0_ratio = (w[-1] - 3 * sigma) / (w[-1] - w[0])
    step_ratio = sigma / (w[-1] - w[0])
    ax.set_yticks((w0_ratio + step_ratio * np.arange(7)) * ny)
    ax.set_yticklabels([r'$-3\sigma$', r'$-2\sigma$', r'$-\sigma$', r'$0$', r'$\sigma$', r'$2\sigma$', r'$3\sigma$'])
    ax.set_xticks([0, nx/4, nx/2, 3*nx/4, nx])
    #ax.set_xticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', r'0', r'$\frac{\pi}{2}$', r'$\pi$'])
    ax.set_xticklabels([r'$-180$', r'$-90$', r'$0$', r'$90$', r'$180$'])
    #ax.vlines(nx/2, ymin=0, ymax=ny, colors='k', linestyles='--')
    ax.set_ylim(w0_ratio * ny, (1 - w0_ratio) * ny)
    ax.set_xlim(0, nx)

    plt.savefig(os.path.join(fig_path, fname), format='pdf', bbox_inches='tight', dpi=300)
    plt.close()


def plot_simulation_connectivity_continuum(w, theta, sigma, control_weights, W_rec, fig_path, connectivity=None, fname='connectivity_simulation.pdf'):

    delta_theta = theta[:,np.newaxis] - theta[np.newaxis,:]
    delta_theta = np.angle(np.exp(1j * delta_theta))
    delta_theta = delta_theta.flatten()

    N = len(theta)
    w_sim = np.tile(control_weights.squeeze(), N).flatten()
    sim_weights = W_rec.flatten()

    dtheta = theta[1] - theta[0]
    theta_edges = np.hstack((theta, theta[-1] + dtheta)) - 0.5 * dtheta
    dw = w[1] - w[0]
    w_edges = np.hstack((w, w[-1] + dw)) - 0.5 * dw

    ret = stats.binned_statistic_2d(delta_theta, w_sim, sim_weights, 'mean', bins=[theta_edges, w_edges])
    connectivity_sim = ret.statistic.T

    if connectivity is not None:
        ny, nx = connectivity.shape
    else:
        nx = len(theta)

    fig, ax = plt.subplots(1,1, figsize=(2,1.5))
    fig.subplots_adjust(left=0.02, bottom=0.25, right=0.95, top=0.75, wspace=0.05)

    w_thr = 3 * sigma
    ind = np.where(w > w_thr)[0][0]

    if connectivity is not None:
        vmax = np.max(connectivity[ind])
    else:
        vmax = 0.2
    
    im = ax.matshow(connectivity_sim, aspect=len(theta)/len(w), cmap='coolwarm', interpolation='none', vmin=-vmax, vmax=vmax) 
    ax.invert_yaxis()
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.set_ylabel(r'$u_{\mathrm{pre}}$')
    ax.set_xlabel(r'$\Delta \theta = \theta_{\mathrm{post}} - \theta_{\mathrm{pre}}$ [deg]')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Recurrent weight')

    ny = len(w)
    w0_ratio = (w[-1] - 3 * sigma) / (w[-1] - w[0])
    step_ratio = sigma / (w[-1] - w[0])
    ax.set_yticks((w0_ratio + step_ratio * np.arange(7)) * ny)
    ax.set_yticklabels([r'$-3\sigma$', r'$-2\sigma$', r'$-\sigma$', r'$0$', r'$\sigma$', r'$2\sigma$', r'$3\sigma$'])
    ax.set_xticks([0, nx/4, nx/2, 3*nx/4, nx])
    ax.set_xticklabels([r'$-180$', r'$-90$', r'$0$', r'$90$', r'$180$'])
    ax.set_ylim(w0_ratio * ny, (1 - w0_ratio) * ny)
    ax.set_xlim(0, nx)

    plt.savefig(os.path.join(fig_path, fname), format='pdf', bbox_inches='tight', dpi=300)
    plt.close()





# def overlay_connectivity_continuum(w, theta, connectivity, control_weights, W_rec, sigma, fig_path, fname='connectivity.png'):

#     delta_theta = theta[np.newaxis,:] - theta[:,np.newaxis]
#     delta_theta = np.angle(np.exp(1j * delta_theta))
#     delta_theta = delta_theta.flatten()

#     N = len(theta)
#     w_sim = np.tile(control_weights, N).flatten()

#     sim_weights = W_rec.flatten()

#     ny, nx = connectivity.shape

#     fig, ax = plt.subplots(1,1, figsize=(5,4))
#     fig.subplots_adjust(left=0.02, bottom=0.25, right=0.95, top=0.75, wspace=0.05)

#     w_thr = 3 * sigma

#     ind = np.where(w > w_thr)[0][0]
#     vmax = np.max(connectivity[ind])
#     #vmin = np.min(connectivity[ind])
#     im = ax.matshow(connectivity, aspect=len(theta)/len(w), cmap='coolwarm', interpolation='none', vmin=-vmax, vmax=vmax)

#     ax.invert_yaxis()
#     ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
#     ax.set_ylabel('presynaptic input weight [a.u.]', fontsize=11)
#     ax.set_xlabel(r'$\Delta \theta$ [rad]')

#     cbar = fig.colorbar(im, ax=ax)
#     cbar.set_label('synaptic weight [a.u.]', fontsize=11)

#     ny = len(w)
#     w0_ratio = (w[-1] - 3 * sigma) / (w[-1] - w[0])  # only display interval from -3 sigma to 3 sigma
#     step_ratio = sigma / (w[-1] - w[0])  # 1 / number of sigmas that w covers
#     ax.set_yticks((w0_ratio + step_ratio * np.arange(7)) * ny)
#     ax.set_yticklabels([r'$-3\sigma$', r'$-2\sigma$', r'$-\sigma$', r'$0$', r'$\sigma$', r'$2\sigma$', r'$3\sigma$'])
#     ax.set_xticks([0, nx/4, nx/2, 3*nx/4, nx])
#     ax.set_xticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', r'0', r'$\frac{\pi}{2}$', r'$\pi$'])
#     #ax.vlines(nx/2, ymin=0, ymax=ny, colors='k', linestyles='--')
#     ax.set_ylim(w0_ratio * ny, (1 - w0_ratio) * ny)
#     ax.set_xlim(0, nx)

#     ax.scatter(nx * (0.5 + delta_theta / (2 * np.pi)), ny * (0.5 + w_sim / (w[-1] - w[0])) , c=sim_weights, cmap='coolwarm', vmin=-vmax, vmax=vmax)

#     plt.savefig(os.path.join(fig_path, fname), format='png', bbox_inches='tight', dpi=300)  #, optimize=True)
#     plt.close()