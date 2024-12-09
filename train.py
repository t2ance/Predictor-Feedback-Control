import argparse
import time
import warnings
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import pickle
import os
import config
from config import DatasetConfig, ModelConfig, TrainConfig
from dynamic_systems import model_forward, DynamicSystem, solve_integral
from plot_utils import plot_result, difference
from utils import pad_zeros, check_dir, load_lr_scheduler, set_everything, load_model, load_optimizer, print_args, \
    SimulationResult, TestResult, count_params, l2s

warnings.filterwarnings('ignore')


def simulation(dataset_config: DatasetConfig, Z0, method=None, model=None, img_save_path: str = None):
    if img_save_path is not None:
        check_dir(img_save_path)

    system: DynamicSystem = dataset_config.system
    ts = dataset_config.ts
    dt = dataset_config.dt
    Z0 = np.array(Z0)
    n_point = dataset_config.n_point
    n_point_delay = dataset_config.n_point_delay
    U = np.zeros((n_point, system.n_input))
    Z = np.zeros((n_point, system.n_state))
    P_numerical_n_iters = np.zeros(n_point)
    P_explicit = np.zeros((n_point, system.n_state))
    P_numerical = np.zeros((n_point, system.n_state))
    P_no = np.zeros((n_point, system.n_state))

    Z[:n_point_delay + 1, :] = Z0
    runtime = 0.

    bar = range(dataset_config.n_point)

    for t_i in bar:
        t = ts[t_i]
        if t_i < n_point_delay:
            t_i_delayed = 0
        else:
            t_i_delayed = t_i - dataset_config.n_point_delay

        Z_t = Z[t_i, :] + dataset_config.noise()

        if method == 'numerical':
            begin = time.time()
            solution = solve_integral(Z_t=Z_t, P_D=P_numerical[t_i_delayed:t_i], U_D=U[t_i_delayed:t_i], t=t,
                                      dataset_config=dataset_config)
            P_numerical[t_i, :] = solution.solution
            P_numerical_n_iters[t_i] = solution.n_iter
            end = time.time()
            runtime += end - begin
            U[t_i] = system.kappa(P_numerical[t_i, :], t)
        elif method == 'no':
            U_D = pad_zeros(segment=U[t_i_delayed:t_i], length=n_point_delay)
            begin = time.time()
            P_no[t_i, :] = model_forward(model=model, U_D=U_D, Z_t=Z_t, t=t)
            end = time.time()
            U[t_i] = system.kappa(P_no[t_i, :], t)
            runtime += end - begin
        else:
            raise NotImplementedError()

        if t_i < n_point_delay:
            U[t_i] = 0

        if n_point_delay <= t_i < len(Z) - 1:
            Z[t_i + 1] = Z[t_i] + dt * system.dynamic(Z[t_i], ts[t_i], U[t_i_delayed])

    plot_result(dataset_config, img_save_path, P_no, P_numerical, Z, U, method)

    D_no = difference(Z, P_no, dataset_config.n_point_delay)
    D_numerical = difference(Z, P_numerical, dataset_config.n_point_delay)
    if method == 'no':
        P = P_no
    elif method == 'numerical':
        P = P_numerical
    else:
        raise NotImplementedError()

    l2_value, rl2_value = l2s(P, Z, dataset_config.n_point_delay)

    success = not (np.any(np.isnan(Z)) or np.any(np.isinf(Z)))
    return SimulationResult(
        Z0=Z0, U=U, Z=Z, D_no=D_no, D_numerical=D_numerical, P_no=P_no, P_numerical=P_numerical,
        runtime=runtime, P_numerical_n_iters=P_numerical_n_iters, avg_prediction_time=runtime / n_point,
        l2=l2_value, rl2=rl2_value, success=success,
        n_parameter=count_params(model) if model is not None else 'N/A')


def result_to_samples(result: SimulationResult, dataset_config):
    n_point_delay = dataset_config.n_point_delay

    inputs = []
    states = []
    predictions = []
    ts = []
    for t_i, t in enumerate(dataset_config.ts):
        if t_i < n_point_delay:
            continue
        t_z_i = t_i - n_point_delay
        t_u_i = t_i - 2 * n_point_delay
        t_z = dataset_config.ts[t_z_i]

        states.append(result.Z[t_z_i])
        prediction = result.Z[t_i - n_point_delay: t_i]
        predictions.append(prediction)
        inputs.append(pad_zeros(result.U[t_u_i:t_z_i], n_point_delay))
        ts.append(t_z)

    samples = []
    for z_pred, z, t, u in zip(predictions, states, ts, inputs):
        def sample_to_tensor(z_features, u_features, time_step_position):
            if z_features is not torch.Tensor:
                z_features = torch.tensor(z_features)
            if u_features is not torch.Tensor:
                u_features = torch.tensor(u_features)
            features = torch.cat((torch.tensor(time_step_position).view(-1), z_features, u_features.view(-1)))
            return features

        samples.append({
            't': torch.tensor(t),
            'z': torch.from_numpy(z),
            'u': torch.from_numpy(u),
            'label': torch.from_numpy(z_pred),
            'input': sample_to_tensor(z, u, t.reshape(-1)),
        })
    return samples


def to_batched_data(batch, device='cuda'):
    return {
        't': torch.stack([sample['t'] for sample in batch]).to(dtype=torch.float32, device=device),
        'z': torch.stack([sample['z'] for sample in batch]).to(dtype=torch.float32, device=device),
        'u': torch.stack([sample['u'] for sample in batch]).to(dtype=torch.float32, device=device),
        'label': torch.stack([sample['label'] for sample in batch]).to(dtype=torch.float32, device=device),
        'input': torch.stack([sample['input'] for sample in batch]).to(dtype=torch.float32, device=device)
    }


def create_simulation_result(dataset_config: DatasetConfig, n_dataset: int = None, test_points=None):
    results = []
    if test_points is None:
        test_points = dataset_config.get_initial_points(n_point=n_dataset)

    print(f'Creating simulation results with {len(test_points)} test points')
    print('Sample test point', test_points[0])
    times = []
    for dataset_idx, Z0 in enumerate(test_points):
        print('dataset_dix', dataset_idx)
        result = simulation(dataset_config, Z0, 'numerical')
        results.append(result)
        times.append(result.avg_prediction_time)
        print(f'Numerical simulation result: {result.l2}, {result.rl2}')

    return results


def run_training(model_config: ModelConfig, train_config: TrainConfig, training_dataset, validation_dataset, model):
    device = train_config.device
    batch_size = train_config.batch_size
    img_save_path = model_config.base_path
    print(f'Train all parameters in {model.name()}')
    check_dir(img_save_path)
    optimizer = load_optimizer(model.parameters(), train_config)
    scheduler = load_lr_scheduler(optimizer, train_config)
    model.train()
    print(f'Training trajectories: {len(training_dataset)}, Validation trajectories: {len(validation_dataset)}')
    training_samples = [sample for traj in training_dataset for sample in traj]
    validation_samples = [sample for traj in validation_dataset for sample in traj]
    print(f'Training samples: {len(training_samples)}, Number of validation samples: {len(validation_samples)}')
    for epoch in range(train_config.n_epoch):
        print(f'Epoch {epoch}')
        np.random.shuffle(training_samples)
        n_iters = 0
        training_loss = 0.0
        for dataset_idx in tqdm(list(range(0, len(training_samples), batch_size))):
            batch = training_samples[dataset_idx:dataset_idx + batch_size]
            optimizer.zero_grad()
            _, loss = model(**to_batched_data(batch, device))
            loss.backward()
            optimizer.step()
            training_loss += loss.detach().item()
            n_iters += 1
        training_loss /= n_iters

        with torch.no_grad():
            n_iters = 0
            validation_loss = 0.0
            for dataset_idx in tqdm(list(range(0, len(validation_samples), batch_size))):
                batch = validation_samples[dataset_idx:dataset_idx + batch_size]
                _, loss = model(**to_batched_data(batch, device))
                validation_loss += loss.detach().item()
                n_iters += 1
            validation_loss /= n_iters
        print(f'Training loss [{training_loss}], Validation loss [{validation_loss}]')
        scheduler.step()

    return model


def run_test(m, dataset_config: DatasetConfig, method: str, base_path: str = None, test_points: List = None):
    begin = time.time()

    base_path = f'{base_path}/{method}'

    l2_list = []
    rl2_list = []
    prediction_time = []
    n_iter_list = []
    results = []
    n_success = 0
    for i, test_point in enumerate(test_points):
        if isinstance(test_point, tuple) and len(test_point) == 2:
            test_point, name = test_point
        else:
            name = None

        if base_path is not None:
            img_save_path = f'{base_path}/{name}'
            check_dir(img_save_path)
        else:
            img_save_path = None
        result = simulation(dataset_config=dataset_config, model=m, Z0=test_point, method=method,
                            img_save_path=img_save_path)
        results.append(result)
        plt.close()

        l2_list.append(result.l2)
        rl2_list.append(result.rl2)
        prediction_time.append(result.avg_prediction_time)
        n_iter_list.append(result.P_numerical_n_iters)
        if result.success:
            n_success += 1
        if not result.success:
            print(f'[WARNING] Running with initial condition Z = {test_point} with method [{method}] failed.')

    l2 = np.mean(l2_list).item()
    rl2 = np.mean(rl2_list).item()
    runtime = np.nanmean(prediction_time).item()
    if method == 'numerical':
        n_iter = np.concatenate(n_iter_list).mean()
        print(f'Numerical method uses {n_iter} iterations on average.')
    end = time.time()
    print(f'Run test time (single test): {end - begin} for method {method}')
    return TestResult(runtime=runtime, rl2=rl2, l2=l2, n_success=n_success, results=results)


def load_dataset(dataset_config):
    if dataset_config.generate:
        print('Begin generating training dataset...')
        training_results = create_simulation_result(
            dataset_config, n_dataset=dataset_config.n_training_dataset)
        print('Begin generating validation dataset...')
        validation_results = create_simulation_result(
            dataset_config, n_dataset=dataset_config.n_validation_dataset)
    else:
        print('Loading data...')

        def read(dataset_dir, split):
            with open(os.path.join(dataset_dir, split + ".pkl"), mode="rb") as file:
                return pickle.load(file)

        training_results = read(f'./data/{dataset_config.system_}', 'training')
        validation_results = read(f'./data/{dataset_config.system_}', 'validation')

    training_dataset = [result_to_samples(result, dataset_config) for result in training_results]
    validation_dataset = [result_to_samples(result, dataset_config) for result in validation_results]

    return training_dataset, validation_dataset


def main(dataset_config: DatasetConfig, model_config: ModelConfig, train_config: TrainConfig):
    set_everything(0)
    test_points = dataset_config.test_points

    training_dataset, validation_dataset = load_dataset(dataset_config)

    model = load_model(train_config, model_config, dataset_config)

    begin = time.time()
    run_training(model_config=model_config, train_config=train_config, training_dataset=training_dataset,
                 validation_dataset=validation_dataset, model=model)
    end = time.time()
    print('Training time', (end - begin))

    test_results = {
        'no': run_test(m=model, dataset_config=dataset_config, base_path=model_config.base_path,
                       test_points=test_points, method='no'),
        'numerical': run_test(m=model, dataset_config=dataset_config, base_path=model_config.base_path,
                              test_points=test_points, method='numerical')
    }

    torch.save(model.state_dict(), f"./{dataset_config.system_}.pth")

    return test_results, model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_name', type=str, default='FNO')

    args = parser.parse_args()
    dataset_config_, model_config_, train_config_ = config.get_config(model_name=args.model_name)

    print_args(dataset_config_)
    print_args(model_config_)
    print_args(train_config_)

    results_, model = main(dataset_config_, model_config_, train_config_)
    for method_, result_ in results_.items():
        print(f'Method: {method_} || L2 error: {result_.l2}'
              f' || RL2 error: {result_.rl2}'
              f' || Prediction time (ms): {result_.runtime * 1000}'
              f' || Successful cases: [{result_.n_success}/{len(dataset_config_.test_points)}]')
