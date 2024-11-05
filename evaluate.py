import numpy as np
import torch
from matplotlib import pyplot as plt

import config
from plot_utils import plot_comparison, plot_difference, plot_control, set_size, fig_width, plot_q
from train import simulation
from utils import set_everything, check_dir, load_model


def figure_bounds(min_, max_):
    interval = max_ - min_
    expanded_interval = interval * 1.1
    new_min = min_ - (expanded_interval - interval) / 2
    new_max = max_ + (expanded_interval - interval) / 2
    return new_min, new_max


def plot_base(plot_name, dataset_config, system, Ps, Zs, Ds, Us, labels, captions, results):
    if system == 'Baxter':
        Ps = [P[:, 4:5] for P in Ps]
        Zs = [Z[:, 4:5] for Z in Zs]
        Ds = [D[:, 4:5] for D in Ds]
        Us = [U[:, 4:5] for U in Us]
        n_row = 4
    elif system == 'Unicycle':
        n_row = 3
    else:
        raise NotImplementedError()

    n_col = len(labels)
    ts = dataset_config.ts
    delay = dataset_config.delay
    n_point_delay = dataset_config.n_point_delay
    fig = plt.figure(figsize=set_size(width=fig_width, subplots=(n_row, n_col)))
    subfigs = fig.subfigures(nrows=1, ncols=n_col)
    method_axes = []

    for subfig, caption in zip(subfigs, captions):
        method_axes.append(subfig.subplots(nrows=n_row, ncols=1, gridspec_kw={'hspace': 0.5}))
        subfig.suptitle(caption)

    P_mins, P_maxs = [], []
    for P in Ps:
        P_mins.append(P.min())
        P_maxs.append(P.max())
    min_p, max_p = figure_bounds(min(*P_mins), max(*P_maxs))

    D_mins, D_maxs = [], []
    for D in Ds:
        D_mins.append(D.min())
        D_maxs.append(D.max())
    min_d, max_d = figure_bounds(min(*D_mins), max(*D_maxs))

    U_mins, U_maxs = [], []
    for U in Us:
        U_mins.append(U.min())
        U_maxs.append(U.max())

    min_u, max_u = figure_bounds(min(*U_mins), max(*U_maxs))

    Z_mins, Z_maxs = [], []
    for Z in Zs:
        Z_mins.append(Z.min())
        Z_maxs.append(Z.max())
    min_z, max_z = figure_bounds(min(*Z_mins), max(*Z_maxs))

    for i, (axes, P, Z) in enumerate(zip(method_axes, Ps, Zs)):
        comment = i == n_col - 1
        plot_comparison(ts, [P], Z, delay, n_point_delay, None, ylim=[min(min_p, min_z), max(max_p, max_z)],
                        ax=axes[0], set_legend=comment)

    for i, (axes, P, Z, D) in enumerate(zip(method_axes, Ps, Zs, Ds)):
        comment = i == n_col - 1
        plot_difference(ts, [P], Z, n_point_delay, None, ylim=[min_d, max_d], ax=axes[1], set_legend=comment,
                        differences=[D], xlim=[0, dataset_config.duration])

    for i, (axes, P, Z, D, U, result, label) in enumerate(
            zip(method_axes, Ps, Zs, Ds, Us, results, labels)):
        comment = i == n_col - 1
        plot_control(ts, U, None, n_point_delay, ax=axes[2], set_legend=comment, ylim=[min_u, max_u])

    if n_row == 4:
        q_des = np.array([dataset_config.system.q_des(t) for t in ts])
        qs = []
        q_des_s = []
        q_mins = []
        q_maxs = []
        for i, (axes, P, Z, D, U) in enumerate(zip(method_axes, Ps, Zs, Ds, Us)):
            q = q_des - Z[:, :dataset_config.n_state // 2]
            if system == 'Baxter':
                q = q[:, 4:5]
                q_des_ = q_des[:, 4:5]
            else:
                raise NotImplementedError()

            q = q[n_point_delay:]
            qs.append(q)
            q_des_s.append(q_des_)
            q_mins.append(min(q.min(), q_des_.min()))
            q_maxs.append(max(q.max(), q_des_.max()))

        min_q, max_q = figure_bounds(min(*q_mins), max(*q_maxs))
        for i, (axes, P, Z, D, U, q, q_des_) in enumerate(zip(method_axes, Ps, Zs, Ds, Us, qs, q_des_s)):
            comment = i == n_col - 1
            plot_q(ts[n_point_delay:], [q], q_des_[n_point_delay:], None, ax=axes[3], set_legend=comment,
                   ylim=[min_q, max_q])

    check_dir(f'./plots')
    plt.savefig(f'./plots/{plot_name}.pdf')

    results_dict = {k: v for k, v in zip(labels, results)}
    return results_dict


def plot_comparisons(test_point, plot_name, dataset_config, system, model=None):
    Ps = []
    Zs = []
    Ds = []
    Us = []
    labels = []
    results = []

    print(f'Begin simulation {plot_name}, with initial point {test_point}')

    points = np.round(test_point, decimals=2)
    points = [str(point) for point in points]
    points = ','.join(points)
    print(f'Solving system with initial point [{points}].')
    result = simulation(dataset_config=dataset_config, Z0=test_point, method='numerical')
    Ps.append(result.P_numerical)
    Zs.append(result.Z)
    Ds.append(result.D_numerical)
    Us.append(result.U)
    labels.append('Successive \n Approximation')
    results.append(result)
    print('Numerical approximation iteration', result.P_numerical_n_iters.mean())
    print('Numerical L2', result.l2)
    print('Numerical RL2', result.rl2)

    m_result = simulation(dataset_config=dataset_config, Z0=test_point, model=model, method='no')
    Ps.append(m_result.P_no)
    Zs.append(m_result.Z)
    Ds.append(m_result.D_no)
    Us.append(m_result.U)
    labels.append('FNO')
    results.append(m_result)
    print('NO L2', m_result.l2)
    print('NO RL2', m_result.rl2)

    print(f'End simulation {plot_name}')

    return plot_base(plot_name, dataset_config, system, Ps, Zs, Ds, Us, labels, labels, results)


if __name__ == '__main__':
    set_everything(0)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str, default='Baxter')
    args = parser.parse_args()

    dataset_config, model_config, train_config = config.get_config(system_=args.s)
    # Add some noise (or not)
    dataset_config.noise_epsilon = 0.

    model = load_model(train_config, model_config, dataset_config)
    model.load_state_dict(torch.load(f'./{dataset_config.system_}.pth'))

    for i, test_point in enumerate(dataset_config.test_points):
        plot_name = f'{args.s}-{i}'
        print(f'{i}-th test point', test_point)

        result_dict = plot_comparisons(
            test_point, plot_name, dataset_config, system=args.s, model=model
        )
