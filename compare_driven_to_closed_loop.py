import numpy as np
import os
import pandas as pd
import argparse
import config
import utils


def compute_normalized_rmse(s_driven, psi_driven, s_cl, psi_cl):

    if hasattr(psi_cl, "__len__"):

    #corr_dists = np.zeros(len(psi_cl))

        norm_rmse = np.zeros(len(psi_cl)) 

        for k in range(len(psi_cl)):

            delta = psi_cl[k] - psi_driven
            delta = np.angle(np.exp(1j * delta))
            delta = np.abs(delta)

            idx = np.argmin(delta)
            #corr_dists[k] = 1 - np.corrcoef(s_driven[idx], s_cl[k]) [0,1]

            mean_act = np.mean(s_driven[idx])

            norm_rmse[k] = np.sqrt(np.mean((s_driven[idx] - s_cl[k])**2)) / mean_act

        return np.mean(norm_rmse)

    else:

        delta = psi_cl - psi_driven
        delta = np.angle(np.exp(1j * delta))
        delta = np.abs(delta)

        idx = np.argmin(delta)

        mean_act = np.mean(s_driven[idx])

        return np.sqrt(np.mean((s_driven[idx] - s_cl)**2)) / mean_act


def main():
    parser = argparse.ArgumentParser(description='Compare driven network to closed loop.')
    parser.add_argument('--sim', type=int, required=False, default=0, help='simulation')
    parser.add_argument('--alpha', type=float, required=False, default=1e-4, help='regularization parameter')
    parser.add_argument('--vel', type=int, required=False, default=360, help='training velocity')
    args = parser.parse_args()

    data_path = utils.get_data_path(config.data_path, config.params)
    data_path = os.path.join(data_path, 'sim={}'.format(args.sim))

    try:

        s_driven = np.load(os.path.join(data_path, 's_driven.npy'), mmap_mode='r')
        turn_ind = np.load(os.path.join(data_path, 'turn_ind.npy'))
        psi_driven = np.load(os.path.join(data_path, 'psi.npy'))
        vels = np.load(os.path.join(data_path, 'vels.npy'))

        train_set = utils.get_train_set_index_single_vel(args.vel)

        data_path2 = os.path.join(data_path, 'set={}_alpha={:.0e}'.format(train_set, args.alpha))

        # with open(os.path.join(data_path2, 'fit_successful.txt')) as f:
        #     fit_successful = int(f.readlines()[0])

        # if fit_successful:
        #     s_cl = np.load(os.path.join(data_path2, 's_vels.npy'), mmap_mode='r')
        #     psi_cl = np.load(os.path.join(data_path2, 'psi_vels.npy'))

    except FileNotFoundError as e:
        raise FileNotFoundError(
            'Simulation data does not exist: {}'.format(data_path)
        ) from e


    time_points = np.array([0, 100, 200, 500, 1000]) / config.dt
    time_points = time_points.astype(int)
    t_arr = []
    v_arr = []
    sample_arr = []
    normalized_rmse = []
    n_perturbations = 10

    for k in range(n_perturbations):

        s_cl = np.load(os.path.join(data_path2, 'perturbations', 's_{}.npy'.format(k)))
        psi_cl = np.load(os.path.join(data_path2, 'perturbations', 'psi_{}.npy'.format(k)))

        for i, v in enumerate(vels):
            for t in time_points:

            #normalized_rmse[i] = compute_normalized_rmse(s_driven[-turn_ind[i]:,i,:], psi_driven[-turn_ind[i]:,i], s_cl[-turn_ind[i]//skip:,i,:], psi_cl[-turn_ind[i]::skip,i])

                rmse = compute_normalized_rmse(s_driven[-turn_ind[i]:,i,:], psi_driven[-turn_ind[i]:,i], s_cl[t,i,:], psi_cl[t,i])

                v_arr.append(v)
                t_arr.append(t)
                sample_arr.append(k)
                normalized_rmse.append(100 * rmse)

    data = {'sample': sample_arr, 'v': v_arr, 't [s]': np.array(t_arr)*config.dt/1000, 'rmse': normalized_rmse}
    df = pd.DataFrame(data)

    df.to_csv(os.path.join(data_path2, 'normalized_rmse_cl_driven.csv'), index=False) 

    #np.save(os.path.join(data_path2, 'normalized_rmse_cl_driven.npy'), normalized_rmse)


if __name__ == "__main__":
    main()