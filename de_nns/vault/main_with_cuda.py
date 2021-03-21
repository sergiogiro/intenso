import json
import os
import time

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import TypeVar
from torchdiffeq import odeint_adjoint as odeint

T = TypeVar

torch.manual_seed(11111)


def get_from_plz_config(key: str, non_plz_value: T) -> T:
    """
    Get the value of a key from the configuration provided by plz

    Or a given value, if not running with plz

    :param key: the key to get, for instance `input_directory`
    :param non_plz_value: value to use when not running with plz
    :return: the value for the key
    """
    configuration_file = os.environ.get('CONFIGURATION_FILE', None)
    if configuration_file is not None:
        with open(configuration_file) as c:
            config = json.load(c)
        return config[key]
    else:
        return non_plz_value


# Default parameters. We use those parameters also when not running via plz
# (one could read them from the command line instead, etc.)
DEFAULT_PARAMETERS = {
    'epochs': 1000,
    'batch_size': 32,
    'eval_batch_size': 32,
    'learning_rate': 0.01,
    'momentum': 0.5
}


def is_verbose_from_cl_args() -> bool:
    """
    Read verbosity from command-line arguments

    Just to illustrate that you can read command line arguments as well

    :return: `True` iff verbose
    """
    cl_args_parser = argparse.ArgumentParser(
        description='Plz PyTorch Example: digit recognition using MNIST')
    cl_args_parser.add_argument('--verbose',
                                action='store_true',
                                help='Print progress messages')
    cl_args = cl_args_parser.parse_args()
    return cl_args.verbose


def write_measures(measures_directory: str, epoch: int, training_loss: float,
                   evaluation_loss: float):
    with open(os.path.join(measures_directory, f'epoch_{epoch:2d}'), 'w') as f:
        json.dump({'training_loss': training_loss, 'accuracy': evaluation_loss}, f)


def load_numpy_arrays(input_directory, kind):
    with open(os.path.join(input_directory, kind + ".npy"), 'rb') as f:
        data = np.load(f)
    with open(os.path.join(input_directory, kind + "_starts.npy"), 'rb') as f:
        data_starts = np.load(f)
    with open(os.path.join(input_directory, kind + "_ends.npy"), 'rb') as f:
        data_ends = np.load(f)
    return data, data_starts, data_ends

FINAL_TIME = 80
OBSERVABLE_VARIABLES = 2

class NNModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        input_dimension = FINAL_TIME * (OBSERVABLE_VARIABLES + 1)
        self.linear1 = nn.Linear(input_dimension, input_dimension*2)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(input_dimension*2, 8)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.linear3 = nn.Linear(8, 1)
        self.data = None
        self.optimizer = None
        self.timeline = torch.ones([FINAL_TIME, OBSERVABLE_VARIABLES+1]) * (-100)
        self.timeline = self.timeline.to(device)
        self.device = device

    def my_forward(self, x):
        x = x.to(self.device)
        x = x.float()
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        return self.linear3(x)

    def set_data(self, data):
        self.data = data

    def __call__(self, t, y: torch.Tensor):
        t = t.to(self.device)
        y = y.to(self.device)
        observed_variables = torch.from_numpy(self.data[int(t.item())])
        observed_variables.float()
        observed_variables = observed_variables.to(self.device)
        new_timepoint = torch.cat([observed_variables, y])
        self.timeline = torch.cat([self.timeline, new_timepoint.view(1, OBSERVABLE_VARIABLES+1)])
        self.timeline = self.timeline[1:]
        return self.my_forward(
            self.timeline.view(-1)
        )


def forward_and_backward(model, training_data, training_start, training_end):
    loss_func = nn.MSELoss()
    model.train()
    model.set_data(training_data)
    model.optimizer.zero_grad()
    time = torch.Tensor(np.linspace(0, FINAL_TIME, FINAL_TIME+1))
    time = time.to(model.device)
    time.float()
    # y = torch.Tensor([training_start]) + v
    # loss = loss_func(torch.Tensor([training_end]).float(), y)
    # print("ys: ", ys, "size:", ys.size())
    start_tensor = torch.Tensor([training_start])
    start_tensor = start_tensor.to(model.device)
    ys = odeint(model, start_tensor, time, method='euler')
    y = ys[-1]
    end_tensor = torch.Tensor([training_end])
    end_tensor = end_tensor.to(model.device)
    loss = loss_func(end_tensor.float(), y)
    loss.backward()
    norm = nn.utils.clip_grad_norm_(model.parameters(), 0.0000001)
    model.optimizer.step()
    print("T Loss:", loss.item())
    return loss.item()

def forward(model, eval_data, eval_start, eval_end):
    loss_func = nn.MSELoss()
    model.eval()
    model.set_data(eval_data)
    time = torch.Tensor(np.linspace(0, FINAL_TIME, FINAL_TIME+1))
    time = time.to(model.device)
    time.float()
    start_tensor = torch.Tensor([eval_start])
    start_tensor = start_tensor.to(model.device)
    ys = odeint(model, start_tensor, time, method='euler')
    y = ys[-1]
    end_tensor = torch.Tensor([eval_end])
    end_tensor = end_tensor.to(model.device)
    loss = loss_func(end_tensor.float(), y)
    norm = nn.utils.clip_grad_norm_(model.parameters(), 0.0000001)
    print("E Loss:", loss.item())
    return loss.item()


def main():
    print('Pytorch version is:', torch.__version__)
    is_verbose = is_verbose_from_cl_args()

    is_cuda_available = torch.cuda.is_available()

    input_directory = get_from_plz_config('input_directory',
                                          os.path.join('..', 'data', 'de_nns'))
    output_directory = get_from_plz_config('output_directory', 'models')
    parameters = get_from_plz_config('parameters', DEFAULT_PARAMETERS)
    # If some parameters weren't passed, use default values for them
    for p in DEFAULT_PARAMETERS:
        if p not in parameters:
            parameters[p] = DEFAULT_PARAMETERS[p]
    measures_directory = get_from_plz_config('measures_directory', 'measures')
    summary_measures_path = get_from_plz_config(
        'summary_measures_path', os.path.join('measures', 'summary'))

    device = 'cpu' # torch.device('cuda' if is_cuda_available else 'cpu')

    if is_verbose:
        print(f'Using device: {device}')

    (training_array, training_starts, training_ends) = load_numpy_arrays(input_directory, 'training')
    orig_training_array = training_array
    training_array = []
    for i in range(0, len(orig_training_array)):
        training_array.append(orig_training_array[i][:FINAL_TIME+1])
    # training_array = training_array[3:4]
    # training_starts = training_starts[3:4]
    # training_ends = training_ends[3:4]
    training_array = training_array[:10]
    training_starts = training_starts[:10]
    training_ends = training_ends[:10]

    (eval_array, eval_starts, eval_ends) = load_numpy_arrays(input_directory, 'eval')
    orig_eval_array = eval_array
    eval_array = []
    for i in range(0, len(orig_eval_array)):
        eval_array.append(orig_eval_array[i][:FINAL_TIME+1])
    eval_array = eval_array[:10]
    eval_starts = eval_starts[:10]
    eval_ends = eval_ends[:10]
    training_time_start = time.time()

    min_loss = np.inf
    training_loss_at_min = 0
    epoch_at_min = 0
    model = NNModel(device).to(device)
    model.optimizer = optim.Adagrad(model.parameters(), lr=0.00001, weight_decay=10)
    for epoch in range(1, parameters['epochs'] + 1):
        n = len(training_array)
        training_loss = 0
        for i in range(0, n):
            print("Training", i)
            training_loss += forward_and_backward(model, training_array[i], training_starts[i], training_ends[i])
        training_loss /= n
        print("Training loss:", training_loss)

        n_eval = len(eval_array)
        eval_loss = 0
        for i in range(0, n_eval):
            print("Eval", i)
            eval_loss += forward(model, eval_array[i], eval_starts[i], eval_ends[i])
        eval_loss /= n_eval

        if is_verbose:
            print(f'Epoch: {epoch}. Training loss: {training_loss:.6f}')
            print(f'Evaluation loss: {eval_loss:.2f} '
                  f'(min loss {min_loss:.2f})')
            print(f'Training time:', time.time() - training_time_start)

        write_measures(measures_directory,
                       epoch=epoch,
                       training_loss=training_loss,
                       evaluation_loss=eval_loss)

        if eval_loss < min_loss:
            min_loss = eval_loss
            training_loss_at_min = training_loss
            epoch_at_min = epoch
            print(f'Best model found at epoch {epoch}, '
                  f'with accuracy {eval_loss:.2f}')
            torch.save(model.state_dict(),
                       os.path.join(output_directory, 'net.pth'))

    with open(summary_measures_path, 'w') as f:
        json.dump(
            {
                'min_loss': min_loss,
                'training_loss_at_min': training_loss_at_min,
                'epoch_at_min': epoch_at_min,
                'training_time': time.time() - training_time_start
            }, f)


if __name__ == '__main__':
    main()
