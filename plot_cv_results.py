import numpy as np
import config
import matplotlib_config
import matplotlib.pyplot as plt
import utils
import os


def main():

    network_params = utils.network_params_str(config.params)

    path1 = utils.get_data_path(config.data_path, config.params)
    if not os.path.exists(path1):
        print('Data missing for {}'.format(path1))
        return False

    i = 0

    path2 = os.path.join(path1, 'sim={}'.format(i))
    if not os.path.exists(path2):
        print('Simulation missing!')
        return False

    training_vels_lst = utils.load_training_vels()

    for k, training_vels in enumerate(training_vels_lst):

        cv_path = os.path.join(path2, 'set={}_cv'.format(k))
        
        alphas = np.load(os.path.join(cv_path, 'alpha_values.npy'))
        cv_values = np.load(os.path.join(cv_path, 'cv_values.npy'))

        with open(os.path.join(cv_path, 'alpha_opt.txt')) as f:
            alpha_opt = float(f.readlines()[0])

        with open(os.path.join(cv_path, 'best_score.txt')) as f:
            best_score = float(f.readlines()[0])

        fig_path = os.path.join('figs', 'cv', network_params, 'sim={}'.format(i))
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)

        fig, ax = plt.subplots(1,1, figsize=(5,3), sharey=True)
        fig.tight_layout()
        ax.plot(alphas, np.mean(cv_values, axis=(0,1)), marker='o')
        ax.plot([alpha_opt], [-best_score], color='r', marker='o')
        ax.set_xscale('log')
        ax.set_yscale('log')

        plt.savefig(os.path.join(fig_path, 'cv_results_set={}.pdf'.format(k)), dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    main()