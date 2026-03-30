import numpy as np
import os
import matplotlib.pyplot as plt
import colorcet as cc
import matplotlib_config


def _plot_act_over_time(t, s, fig_path, fname='population_act_over_time.pdf', xticks=None, minutes=False):

    fig = plt.figure(figsize=(2.25,1.1))
    ax = fig.add_axes([0.225, 0.35, 0.55, 0.6]) 
    im = ax.imshow(s.T, extent=[t[0]/1000, t[-1]/1000, -np.pi, np.pi], aspect='auto', vmin=0, vmax=s.max(), origin='lower', interpolation='none', cmap=cc.cm.fire);
    if xticks is not None:
        ax.set_xticks(xticks)
    #ax.set_yticks([1,1000])
    ax.set_yticks([-np.pi, 0, np.pi], labels=[-180, 0, 180]);

    if minutes:
        ax.set_xticklabels((np.array(xticks) / 60).astype(int))
        ax.set_xlabel(r'$t$ [min]')
    else:
        ax.set_xlabel(r'$t$ [s]')

    #ax.set_ylabel('Neuron #')
    ax.set_ylabel(r'$\theta$ [deg]')
    #cb=plt.colorbar(im, ax=ax, fraction=0.05, pad=0.04)

    cax = fig.add_axes([0.8, 0.4, 0.015, 0.475])
    cb=plt.colorbar(im, cax=cax)

    cb.set_label('Activation [a.u.]')
    plt.savefig(os.path.join(fig_path, fname), format='pdf', dpi=300)
    plt.close()


def _plot_act_over_time_doublering(t, s, fig_path, fname='population_act_over_time.pdf', xticks=None, minutes=False):

    fig = plt.figure(figsize=(2.25,1.1))

    for i in range(2):

        if i == 0:
            ax = fig.add_axes([0.225, 0.675, 0.55, 0.3]) 
            im = ax.imshow(s[:,:N//2].T, extent=[t[0]/1000, t[-1]/1000, -np.pi, np.pi], aspect='auto', vmin=0, vmax=s.max(), origin='lower', interpolation='none', cmap=cc.cm.fire);

            if xticks is not None:
                ax.set_xticks(xticks)
            else:
                ax.set_xticks(np.arange(1+np.ceil(t[-1]/1000)))
            ax.set_xticklabels([])

        else:
            ax = fig.add_axes([0.225, 0.3, 0.55, 0.3]) 
            im = ax.imshow(s[:,N//2:].T, extent=[t[0]/1000, t[-1]/1000, -np.pi, np.pi], aspect='auto', vmin=0, vmax=s.max(), origin='lower', interpolation='none', cmap=cc.cm.fire);

            if xticks is not None:
                ax.set_xticks(xticks)
            else:
                ax.set_xticks(np.arange(1+np.ceil(t[-1]/1000)))

            if minutes:
                ax.set_xticklabels(np.array(xticks).astype(int) / 60)
                ax.set_xlabel(r'$t$ [min]')
            else:
                ax.set_xlabel(r'$t$ [s]')

        ax.set_yticks([-np.pi, 0, np.pi], labels=[-180, 0, 180]);
        ax.set_ylabel(r'$\theta$ [deg]')

    cax = fig.add_axes([0.8, 0.4, 0.015, 0.475])
    cb=plt.colorbar(im, cax=cax)
    cb.set_label('Activation [a.u.]')

    plt.savefig(os.path.join(fig_path, fname), format='pdf', dpi=300)
    plt.close()


def plot_act_over_time(t, s, fig_path, fname='population_act_over_time.pdf', xticks=None, minutes=False, double_ring=False):
    if double_ring:
        _plot_act_over_time_doublering(t, s, fig_path, fname=fname, xticks=xticks, minutes=minutes)
    else:
        _plot_act_over_time(t, s, fig_path, fname=fname, xticks=xticks, minutes=minutes)


def plot_bump_shapes_over_time(theta, s, fig_path, fname='bump_shapes.pdf'):
    N = len(s[0])
    fig = plt.figure(figsize=(2.25,1.1))
    ax = fig.add_axes([0.2, 0.35, 0.55, 0.6]) 

    cmap = plt.get_cmap('copper_r')
    colors = cmap(np.linspace(0,1,len(s)))

    for bump_shape, c in zip(s, colors):
        im = ax.scatter(theta, bump_shape, marker='.', edgecolors='none', s=3, color=c, alpha=1.0)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], labels=[-180, -90, 0, 90, 180]);

    ax.set(xlabel=r'$\theta$ [deg]', ylabel='')
    ax.set_ylabel('Activation [a.u.]')
    ax.set_ylim(0, 1.1*np.array(bump_shape).max())

    plt.savefig(os.path.join(fig_path, fname), dpi=300)
    plt.close()
