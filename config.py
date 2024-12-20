from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np

import dynamic_systems

root_dir = '.'


@dataclass
class ModelConfig:
    fno_n_modes_height: Optional[int] = field(default=16)
    fno_hidden_channels: Optional[int] = field(default=32)
    fno_n_layer: Optional[int] = field(default=4)
    deeponet_hidden_size: Optional[int] = field(default=64)
    deeponet_n_layer: Optional[int] = field(default=5)

    gru_n_layer: Optional[int] = field(default=4)
    gru_hidden_size: Optional[int] = field(default=64)
    lstm_n_layer: Optional[int] = field(default=4)
    lstm_hidden_size: Optional[int] = field(default=64)

    model_name: Optional[str] = field(default='FNO')
    system: Optional[str] = field(default='Baxter')

    @property
    def base_path(self):
        return f'{root_dir}/{self.system}/result'


@dataclass
class TrainConfig:
    batch_size: Optional[int] = field(default=64)
    learning_rate: Optional[float] = field(default=1e-4)
    weight_decay: Optional[float] = field(default=.0)
    n_epoch: Optional[int] = field(default=100)
    device: Optional[str] = field(default='cuda')
    scheduler_step_size: Optional[int] = field(default=1)
    scheduler_gamma: Optional[float] = field(default=.99)
    system: Optional[str] = field(default='Baxter')

    @property
    def model_save_path(self):
        return f'{root_dir}/{self.system}/checkpoint'


@dataclass
class DatasetConfig:
    delay: float = field(default=0)
    duration: Optional[int] = field(default=8)
    dt: Optional[float] = field(default=0.125)

    generate: Optional[bool] = field(default=False)
    successive_approximation_threshold: Optional[float] = field(default=1e-7)

    ic_lower_bound: Optional[float] = field(default=-2)
    ic_upper_bound: Optional[float] = field(default=2)
    n_training_dataset: Optional[int] = field(default=200)
    n_validation_dataset: Optional[int] = field(default=200)

    noise_epsilon: Optional[float] = field(default=0.)
    baxter_dof: Optional[int] = field(default=2)
    baxter_f: Optional[float] = field(default=0.1)
    baxter_alpha: Optional[float] = field(default=1)
    baxter_beta: Optional[float] = field(default=1)
    baxter_magnitude: Optional[float] = field(default=0.2)
    scheduler_step_size: Optional[int] = field(default=1)
    scheduler_gamma: Optional[float] = field(default=1.)

    n_test_point: int = field(default=25)
    system_: Optional[str] = field(default='Baxter')

    random_test_lower_bound: Optional[float] = field(default=0.)
    random_test_upper_bound: Optional[float] = field(default=0.)

    @property
    def base_path(self):
        return f'{root_dir}/{self.system_}/datasets'

    def get_initial_points(self, n_point=1):
        state = np.random.RandomState(seed=0)
        return [
            tuple((state.uniform(self.random_test_lower_bound, self.random_test_upper_bound,
                                 self.system.n_state)).tolist()) for _ in range(n_point)
        ]

    @property
    def test_points(self) -> List[Tuple]:
        return self.get_initial_points(self.n_test_point)

    @property
    def n_state(self):
        return self.system.n_state

    @property
    def system(self):
        if self.system_ == 'Baxter':
            return dynamic_systems.Baxter(alpha=self.baxter_alpha, beta=self.baxter_beta, dof=self.baxter_dof,
                                          f=self.baxter_f, magnitude=self.baxter_magnitude)
        else:
            raise NotImplementedError('Unknown system', self.system_)

    @property
    def ts(self) -> np.ndarray:
        return np.linspace(-self.delay, self.duration - self.dt, self.n_point)

    @property
    def n_point(self) -> int:
        return self.n_point_delay + self.n_point_duration

    @property
    def n_point_delay(self) -> int:
        return int(round(self.delay / self.dt))

    @property
    def n_point_duration(self) -> int:
        return int(round(self.duration / self.dt))

    def noise(self):
        if self.noise_epsilon == 0:
            return 0
        return np.random.randn() * self.noise_epsilon


def get_config(model_name):
    dataset_config = DatasetConfig(
        system_='Baxter', delay=0.5, duration=8, dt=0.02, n_training_dataset=600, n_validation_dataset=100,
        baxter_dof=5, baxter_f=1, baxter_magnitude=0.1, baxter_alpha=1, baxter_beta=2, ic_lower_bound=0,
        ic_upper_bound=1, random_test_lower_bound=0, random_test_upper_bound=1)
    if model_name == 'FNO':
        model_config = ModelConfig(fno_n_layer=2, fno_hidden_channels=128, fno_n_modes_height=20)
        train_config = TrainConfig(n_epoch=100, learning_rate=3e-3, weight_decay=2e-6, batch_size=4096)
    elif model_name == 'DeepONet':
        model_config = ModelConfig(deeponet_hidden_size=64, deeponet_n_layer=5)
        train_config = TrainConfig(n_epoch=100, learning_rate=3e-3, weight_decay=1.5e-2, batch_size=4096)
    elif model_name == 'GRU':
        model_config = ModelConfig(gru_n_layer=5, gru_hidden_size=64)
        train_config = TrainConfig(n_epoch=100, learning_rate=9.5e-3, weight_decay=5e-6, batch_size=4096)
    elif model_name == 'LSTM':
        model_config = ModelConfig(lstm_n_layer=5, lstm_hidden_size=64)
        train_config = TrainConfig(n_epoch=100, learning_rate=9e-3, weight_decay=8.5e-4, batch_size=4096)
    elif model_name == 'FNO+GRU':
        model_config = ModelConfig(fno_n_modes_height=24, fno_hidden_channels=128, fno_n_layer=3, gru_n_layer=5,
                                   gru_hidden_size=64)
        train_config = TrainConfig(n_epoch=100, learning_rate=6e-3, weight_decay=2e-4, batch_size=4096)
    elif model_name == 'DeepONet+GRU':
        model_config = ModelConfig(deeponet_hidden_size=128, deeponet_n_layer=3, gru_n_layer=5, gru_hidden_size=64)
        train_config = TrainConfig(n_epoch=100, learning_rate=0.62e-3, weight_decay=8e-2, batch_size=4096)
    else:
        raise NotImplementedError(f'Unknown model: {model_name}')
    model_config.model_name = model_name
    return dataset_config, model_config, train_config


if __name__ == '__main__':
    ...
