import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from optuna import Trial
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from util import PersonalDataLoader, DEVICE

# for extra speed
torch.backends.cudnn.benchmark = True

DIR = 'testing2.db'


# network definition, takes in hidden dim and activation
class Model(nn.Module):

    def __init__(self, hidden_dim, activation):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, hidden_dim),
            activation,
            nn.Linear(hidden_dim, 10)
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.layers.forward(x)


def objective(trial: Trial):
    """ objective function to be minimized """
    ### generate hyperparameters.
    activation, batch_size, hidden_dim, lr, weight_decay = get_params(trial)
    print(
        "STARTING TRAIL",
        dict(
            lr=lr,
            weight_decay=weight_decay,
            activation=activation,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
        )
    )

    ### run training
    stats = train(activation, batch_size, hidden_dim, lr, weight_decay)

    ### determine final metric that indicates the value (here just the max accuracy)
    objective = max(stats)

    print("FINISHED TRIAL", objective)

    return objective  # An objective value linked with the Trial object.


def get_params(trial):
    """ defines hyperparams and ranges to optimize """
    lr = trial.suggest_loguniform(
        'lr',
        low=1e-5,
        high=1e-1
    )
    weight_decay = trial.suggest_loguniform(
        'weight_decay',
        low=1e-6,
        high=1e-2
    )
    activation = trial.suggest_categorical(
        'activation',
        choices=['relu', 'sigmoid', 'leakyrelu', 'tanh']
    )
    hidden_dim = 2 ** trial.suggest_int("hidden_dim", low=2, high=9)
    batch_size = 2 ** trial.suggest_int("batch_size", low=4, high=11)
    return activation, batch_size, hidden_dim, lr, weight_decay


def train(activation, batch_size, hidden_dim, lr, weight_decay):
    """ trains a network """

    # get model
    activation = \
        dict(
            relu=nn.ReLU(),
            sigmoid=nn.Sigmoid(),
            leakyrelu=nn.LeakyReLU(negative_slope=0.05),
            tanh=nn.Tanh()
        )[activation]  # convert string to actual activation
    network = Model(hidden_dim=hidden_dim, activation=activation).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.SGD(network.parameters(), lr=lr, weight_decay=weight_decay)

    # get data
    train_data = PersonalDataLoader(train=True, batch_size=batch_size)
    test_data = PersonalDataLoader(train=False, batch_size=batch_size)

    # for extra speed
    scaler = GradScaler()
    cuda_batch_size = torch.cuda.FloatTensor([batch_size], device=DEVICE)
    cuda_test_len = torch.cuda.FloatTensor([len(test_data)], device=DEVICE)

    # do training
    epoch_acc = []
    for epoch in range(3):
        # > train
        network.train()
        for x, y in tqdm(train_data):
            loss = forward_pass(loss_fn, network, x, y)
            backward_pass(loss, opt, scaler)
        # > test
        with torch.no_grad():
            network.eval()
            aggr = torch.cuda.FloatTensor([0], device=DEVICE)
            for x, y in tqdm(test_data):
                aggr += get_accuracy_batch(cuda_batch_size, network, x, y)
            epoch_acc.append((aggr / cuda_test_len).detach().cpu().item())
    return epoch_acc


def get_accuracy_batch(cuda_batch_size, network, x, y):
    """ outputs accuracy of this batch predictions """
    output = network.forward(x)
    predictions = output.argmax(dim=-1, keepdim=True).view_as(y)
    correct = y.eq(predictions).sum()
    return correct / cuda_batch_size


def forward_pass(loss_fn, network, x, y):
    """ does forward pass in training """
    #  quicker than zero_drad()
    for param in network.parameters():
        param.grad = None
    with autocast():
        output = network.forward(x)
        loss = loss_fn.forward(output, y)
    return loss


def backward_pass(loss, opt, scaler):
    """ does backward pass in training """
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()


study = optuna.create_study(
    load_if_exists=True,
    study_name=DIR.split(".")[0],
    storage=f"sqlite:///{DIR}",
    direction='maximize'
)  # Create a new study.

try:
    print("STARTING TUNING INDEFINITELY\npress ctrl-c to interrupt at any time. the process can be resumed")
    study.optimize(objective)  # Invoke optimization of the objective function.
except KeyboardInterrupt:
    print("Killed by user")
except Exception as e:
    print(f"Something went wrong in tuning ({e.__class__}): {e}")

df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
print(df)
