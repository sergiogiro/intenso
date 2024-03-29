import argparse
import json
import os
import time
from typing import TypeVar

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

T = TypeVar

torch.manual_seed(11111)

"""
BOILERPLATE FOR PLZ

You can skim until END OF BOILERPLATE FOR PLZ
"""


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
    'epochs': 10000,
    'batch_size': 1000,
    'eval_batch_size': 100,
    'learning_rate': 0.001,
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
                   evaluation_loss: float, best_epoch: int,
                   best_evaluation_loss: float,
                   training_as_eval_loss:float):
    with open(os.path.join(measures_directory, f'epoch_{epoch:05d}'), 'w') as f:
        json.dump({'training_loss': training_loss, 'accuracy': evaluation_loss,
                   'best_epoch': best_epoch,
                   'best_evaluation_loss': best_evaluation_loss,
                   'training_as_eval_loss': training_as_eval_loss}, f)


"""
END OF BOILERPLATE FROM PLZ
"""


def load_numpy_arrays(input_directory, kind):
    """
    Load the arrays generated in the notebook
    """
    with open(os.path.join(input_directory, kind + ".npy"), 'rb') as f:
        # noinspection PyTypeChecker
        data = np.load(f)
    with open(os.path.join(input_directory, kind + "_starts.npy"), 'rb') as f:
        # noinspection PyTypeChecker
        data_starts = np.load(f)
    with open(os.path.join(input_directory, kind + "_ends.npy"), 'rb') as f:
        # noinspection PyTypeChecker
        data_ends = np.load(f)
    with open(os.path.join(input_directory, kind + "_ys.npy"), 'rb') as f:
        # noinspection PyTypeChecker
        data_ys = np.load(f)
    with open(os.path.join(input_directory, kind + "_ips.npy"), 'rb') as f:
        # noinspection PyTypeChecker
        data_ips = np.load(f)
    return data, data_starts, data_ends, data_ys, data_ips


FINAL_TIME = 19
OBSERVABLE_VARIABLES = 3
NUM_SPARSE_VARS = 1

class NNModel(nn.Module):
    """
    Neural network. Just three layers with relu activation on each, and dropout
    before the last one.

    See comments to __call__ for details
    """

    def __init__(self, device: str):
        super().__init__()
        input_dimension = (FINAL_TIME + 1) * (OBSERVABLE_VARIABLES + NUM_SPARSE_VARS)
        self.linear1 = nn.Linear(input_dimension, input_dimension * 2)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(input_dimension * 2, 100)
        self.relu2 = nn.ReLU()
        # self.dropout = nn.Dropout(0.1)
        self.linear3 = nn.Linear(100, OBSERVABLE_VARIABLES + NUM_SPARSE_VARS)
        self.optimizer = None
        self.device = device

    def my_forward(self, x):
        x = x.to(self.device)
        x = x.float()
        x = self.linear1(x)
        x = self.relu1(x)
        # x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu2(x)
        # x = self.dropout(x)
        return self.linear3(x)

    def to_tensor(self, array):
        """
        Make sure the array is a tensor usable for this model.

        If necessary, move it to the right device and set precision to float
        """
        if not isinstance(array, torch.Tensor):
            tensor = torch.Tensor(array)
        else:
            tensor = array
        return tensor.to(self.device).float()

    def __call__(self, timeline):
        self.to_tensor(timeline)
        return self.my_forward(timeline.view(-1))


def euler(model, xs, start_tensor: torch.Tensor, time_tensor: torch.Tensor):
    xs = model.to_tensor(xs)
    timeline = torch.Tensor(xs.size()[0], xs.size()[1] + start_tensor.size()[0])
    timeline.fill_(-100)
    timeline = model.to_tensor(timeline)
    timeline[0][:] = torch.cat([xs[0], start_tensor])
    ys = model.to_tensor(torch.Tensor(xs.size()[0], start_tensor.size()[0]))
    ys[0][:] = start_tensor
    for i in range(1, xs.size()[0]):
        time_delta = (time_tensor[i] - time_tensor[i-1]).item()
        y_prime = model(timeline.clone().detach())[:NUM_SPARSE_VARS]
        ys[i][:] = ys[i-1][:] + time_delta * y_prime
        timeline[i][:xs.size()[1]] = xs[i]
        timeline[i][xs.size()[1]:] = ys[i]
    return ys


def forward(model, training_data, training_start, training_end, training_ys,
            training_ips, and_backward):
    """
    Method to perform one training step (iff and_backward) or an evaluation step

    Returns the loss
    """
    loss_func = nn.MSELoss()
    if and_backward:
        model.train()
        model.optimizer.zero_grad()
    else:
        model.eval()
    time_tensor = model.to_tensor(np.linspace(0, FINAL_TIME, FINAL_TIME + 1))
    start_tensor = model.to_tensor([training_start])
    ys = euler(model, training_data, start_tensor, time_tensor)
    loss = loss_func(model.to_tensor([training_end]), ys[-1])
    for p in training_ips:
        loss += loss_func(model.to_tensor([training_ys[p]]), ys[p])
    #loss += 100 * torch.sum(nn.ReLU()(-ys))
    # loss += 100 * torch.sum(nn.ReLU()(ys-100))
    print("Loss:", loss.item())
    if and_backward:
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        model.optimizer.step()
    return loss.item()


def load_model(device, input_directory, learning_rate):
    model = NNModel(device).to(device)
    warmup_model_name = os.path.join(input_directory, 'net.pth')
    if os.path.isfile(warmup_model_name):
        print('Using warmup model!')
        model.load_state_dict(
            torch.load(warmup_model_name, map_location=torch.device(device))
        )
    model.optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    return model


def do_epoch(model, training_array, training_starts, training_ends, training_ys,
             training_ips, is_training):
    n = len(training_array)
    training_loss = 0
    for i in range(0, n):
        print("Training" if is_training else "Eval", i)
        training_loss += forward(model, training_array[i], training_starts[i],
                                 training_ends[i], training_ys[i],
                                 training_ips[i], and_backward=is_training)
    training_loss /= n
    print("Loss:", training_loss)
    return training_loss


def retrieve_values(input_directory, limit, is_training=False):
    (training_array, training_starts, training_ends, training_ys,
     training_ips) = load_numpy_arrays(input_directory,
                                       'training' if is_training else 'eval')
    training_array = training_array[:limit]
    return training_array, training_starts, training_ends, training_ys, training_ips


def main():
    print('Pytorch version is:', torch.__version__)
    is_verbose = is_verbose_from_cl_args()

    is_cuda_available = torch.cuda.is_available()

    input_directory = get_from_plz_config('input_directory',
                                          os.path.join('..', 'data', 'de_nns'))
    output_directory = get_from_plz_config('output_directory', 'models')
    try:
        mlflow.get_experiment_by_name("de_nns")
    except Exception as e:
        print("Raised:" + str(type(e)))
        mlflow.create_experiment("de_nns")
    mlflow.set_experiment("de_nns")
    parameters = get_from_plz_config('parameters', DEFAULT_PARAMETERS)
    mlflow.log_params(parameters)
    # If some parameters weren't passed, use default values for them
    for p in DEFAULT_PARAMETERS:
        if p not in parameters:
            parameters[p] = DEFAULT_PARAMETERS[p]
    measures_directory = get_from_plz_config('measures_directory', 'measures')
    summary_measures_path = get_from_plz_config(
        'summary_measures_path', os.path.join('measures', 'summary'))

    # Use always cpu to make it more portable
    device = torch.device('cuda' if is_cuda_available else 'cpu')

    if is_verbose:
        print(f'Using device: {device}')

    training_array, training_starts, training_ends, training_ys, training_ips = retrieve_values(
        input_directory,
        limit=parameters['batch_size'],
        is_training=True)
    eval_array, eval_starts, eval_ends, eval_ys, eval_ips = retrieve_values(
        input_directory, limit=parameters['eval_batch_size'],
        is_training=False)

    training_time_start = time.time()

    min_loss = np.inf
    training_loss_at_min = 0
    epoch_at_min = 0
    model = load_model(device, input_directory,
                       learning_rate=parameters['learning_rate'])
    for epoch in range(1, parameters['epochs'] + 1):
        if epoch != 1:
            training_loss = do_epoch(
                model, training_array, training_starts, training_ends,
                training_ys, training_ips, is_training=True
            )
        else:
            training_loss = np.inf

        training_as_eval_loss = do_epoch(
            model, training_array[:len(eval_array)], training_starts, training_ends,
            training_ys, training_ips, is_training=False
        )


        eval_loss = do_epoch(
            model, eval_array, eval_starts, eval_ends, eval_ys, eval_ips,
            is_training=False
        )



        if is_verbose:
            print(f'Epoch: {epoch}. Training loss: {training_loss:.6f}')
            print(f'Evaluation loss: {eval_loss:.2f} '
                  f'(min loss {min_loss:.2f})')
            print(f'Training time:', time.time() - training_time_start)

        torch.save(model.state_dict(),
                   os.path.join(output_directory, f'net_{epoch:05d}.pth'))

        # If model is best save with a special name and remember the epoch
        if eval_loss < min_loss:
            min_loss = eval_loss
            training_loss_at_min = training_loss
            epoch_at_min = epoch
            print(f'Best model found at epoch {epoch}, '
                  f'with accuracy {eval_loss:.2f}')
            torch.save(model.state_dict(),
                       os.path.join(output_directory, 'net.pth'))

        write_measures(measures_directory,
                       epoch=epoch,
                       training_loss=training_loss,
                       evaluation_loss=eval_loss,
                       best_epoch=epoch_at_min,
                       best_evaluation_loss=min_loss,
                       training_as_eval_loss=training_as_eval_loss)

    metrics = {
        'min_loss': min_loss,
        'training_loss_at_min': training_loss_at_min,
        'epoch_at_min': epoch_at_min,
        'training_time': time.time() - training_time_start
    }
    with open(summary_measures_path, 'w') as f:
        json.dump(metrics, f)
    mlflow.log_metrics(metrics)


if __name__ == '__main__':
    main()
