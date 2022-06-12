from notmnist import load_notmnist
import torch.nn.functional as F
import torch

letters = 'ABCDEFGHIJ'
X_train, y_train, X_test, y_test = load_notmnist(letters=letters)
X_train, X_test = X_train.reshape([-1, 784]), X_test.reshape([-1, 784])

X_train = torch.tensor(X_train.T)
y_train = F.one_hot(torch.tensor(y_train)).T
X_test = torch.tensor(X_test.T)
y_test = F.one_hot(torch.tensor(y_test)).T

import random # noqa

random_index = random.randrange(0, X_train.shape[1])


def tanh(x):
    return x.tanh()


def relu(x):
    return x.relu()


def softmax(x):
    return x.softmax(dim=0)


def derative_tanh(x):
    return 1 - x ** 2


def derative_relu(x):
    df = x.clone().detach() > 0
    return df.float()


def initialize_parameters(n_x, n_h, n_y):
    """
      n_x - кол-во нейронов во входном слое
      n_h - кол-во нейронов в скрытом слое
      n_y - кол-во нейронов в выходном слое

    """

    w1 = torch.randn(n_h, n_x) * 0.01
    b1 = torch.zeros(n_h)

    w2 = torch.randn(n_y, n_h) * 0.01
    b2 = torch.zeros(n_h)

    parameters = {
        'w1': w1,
        'b1': b1,
        'w2': w2,
        'b2': b2
    }

    return parameters


def forward_propagation(x, parameters):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    z1 = w1 @ x + b1
    a1 = relu(z1)

    z2 = w2 @ a1 + b2
    a2 = softmax(z2)

    forward_cache = {
        'z1': z1,
        'a1': a1,
        'z2': z2,
        'a2': a2
    }

    return forward_cache


def cost_function(a2, y):
    m = y.shape[1]

    cost = -(1 / m) * torch.sum(y * torch.log(a2))

    return cost


def back_propagation(x, y, parameters, forward_cache):
    w1 = parameters['w1'] # noqa
    b1 = parameters['b1'] # noqa
    w2 = parameters['w2']
    b2 = parameters['b2'] # noqa

    a1 = forward_cache['a1']
    a2 = forward_cache['a2']

    m = y.shape[1]

    dz2 = a2 - y
    dw2 = (1 / m) * dz2 @ a1.T
    db2 = (1 / m) * torch.sum(dz2, axis=1, keepdim=True)

    dz1 = (1 / m) * w2.T @ dz2 * derative_relu(a1)
    dw1 = (1 / m) * dz1 @ x.T
    db1 = (1 / m) * torch.sum(dz1, axis=1, keepdim=True)

    gradients = {
        'dw1': dw1,
        'db1': db1,
        'dw2': dw2,
        'db2': db2
    }

    return gradients


def update_parameters(parameters, gradients, learning_rate):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    dw1 = gradients['dw1']
    db1 = gradients['db1']
    dw2 = gradients['dw2']
    db2 = gradients['db2']

    w1 = w1 - learning_rate * dw1
    b1 = b1 - learning_rate * db1
    w2 = w2 - learning_rate * dw2
    b2 = b2 - learning_rate * db2

    parameters = {
        'w1': w1,
        'b1': b1,
        'w2': w2,
        'b2': b2
    }

    return parameters


def model(x, y, n_h, learning_rate, iterations):
    n_x = x.shape[0]
    n_y = y.shape[0]

    cost_list = []

    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(iterations):

        indexes = random.sample(range(0, X_train.shape[1]), n_h)

        x_samples, y_samples = x[:, indexes], y[:, indexes]

        forward_cache = forward_propagation(x_samples, parameters)

        cost = cost_function(forward_cache['a2'], y_samples)

        gradients = back_propagation(x_samples, y_samples,
                                     parameters, forward_cache)

        parameters = update_parameters(parameters, gradients, learning_rate)

        cost_list.append(cost)

        if i % (iterations / 10) == 0:
            with open("result.txt", "a+") as ouf:
                ouf.write(f"Cost after {i} iterations is: {cost}\n")

    return parameters, cost_list


iterations = 1000
n_h = 30 ** 2
learning_rate = 0.9

Parameters, Cost_list = model(X_train, y_train, n_h, learning_rate, iterations)

t = torch.arange(0, iterations)


def accuracy(inp, labels, parameters):
    forward_cache = forward_propagation(inp, parameters)
    a_out = forward_cache['a2']

    _, a_out = torch.topk(a_out, k=1, dim=0)
    _, y_out = torch.topk(labels, k=1, dim=0)

    acc = float(torch.mean((a_out == y_out).float()) * 100)

    return acc


with open("result.txt", "a+") as ouf:
    ouf.write(
        f'Accuracy of Train Dataset is: {round(accuracy(X_train[:, :n_h], y_train[:, :n_h], Parameters), 2)}%.\n')  # noqa
    ouf.write(
        f'Accuracy of Test Dataset is: {round(accuracy(X_test[:, :n_h], y_test[:, :n_h], Parameters), 2)}%.\n')  # noqa

random_inx = random.randrange(0, X_test.shape[1])

forward_cache = forward_propagation(X_test[:, random_inx].reshape(X_test.shape[0], 1), Parameters) # noqa
a_out = forward_cache['a2']

_, a_out = torch.topk(a_out, k=1, dim=0)

print('Our model says, it is:', letters[int(a_out.squeeze()[0])])

