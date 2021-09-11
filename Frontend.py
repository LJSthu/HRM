import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn


def pretty(vector):
    if type(vector) is list:
        vlist = vector
    elif type(vector) is np.ndarray:
        vlist = vector.reshape(-1).tolist()
    else:
        vlist = vector.view(-1).tolist()
    return "[" + ", ".join("{:+.4f}".format(vi) for vi in vlist) + "]"


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        self.weight_init()

    def weight_init(self):
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


class WeightedLasso:
    def __init__(self, X, y, weight, lam):
        self.model = LinearRegression(X.shape[1], 1)
        self.X = X
        self.y = y
        self.weight = weight.reshape(-1, 1)
        self.loss = nn.MSELoss()
        self.lam = lam
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self):
        self.model.weight_init()
        epochs = 3000

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            pred = self.model(self.X)
            loss = self.loss(pred, self.y) +\
                   self.lam * torch.mean(torch.abs(self.weight*self.model.linear.weight.reshape(self.weight.shape)))
            loss.backward(retain_graph=True)
            self.optimizer.step()
        return self.model.linear.weight.clone().cpu().detach(), self.model.linear.bias.clone().cpu().detach()



class McModel:
    def __init__(self, num_classes, X, y):
        self.num_classes = num_classes
        self.X = X
        self.y = y.reshape(-1, 1)
        self.center = None
        self.bias = None
        self.domain = None
        self.weights = None

    def ols(self):
        for i in range(self.num_classes):
            index = torch.where(self.domain == i)[0]
            tempx = (self.X[index, :]).reshape(-1, self.X.shape[1])
            tempy = (self.y[index, :]).reshape(-1, 1)
            clf = WeightedLasso(tempx, tempy, self.weights, 1.0)
            self.center[i, :], self.bias[i] = clf.train()

    def cluster(self, weight, past_domains, reuse=False):
        self.center = torch.tensor(np.zeros((self.num_classes, self.X.shape[1]), dtype=np.float32))
        self.bias = torch.tensor(np.zeros(self.num_classes, dtype=np.float32))

        if past_domains is None or not reuse:
            self.domain = torch.tensor(np.random.randint(0, self.num_classes, self.X.shape[0]))
        else:
            self.domain = past_domains
        assert self.domain.shape[0] == self.X.shape[0]
        self.weights = weight

        iter = 0
        end_flag = False
        delta_threshold = 250

        while not end_flag:
            iter += 1
            self.ols()
            ols_error = []
            for i in range(self.num_classes):
                coef = self.center[i].reshape(-1, 1)
                error = torch.abs(torch.mm(self.X, coef) + self.bias[i] - self.y)
                assert error.shape == (self.X.shape[0], 1)
                ols_error.append(error)
            ols_error = torch.stack(ols_error, dim=0).reshape(self.num_classes, self.X.shape[0])
            new_domain = torch.argmin(ols_error, dim=0)
            assert new_domain.shape[0] == self.X.shape[0]
            diff = self.domain.reshape(-1, 1) - new_domain.reshape(-1, 1)
            diff[diff != 0] = 1
            delta = torch.sum(diff)
            if iter % 10 == 9:
                print("Iter %d | Delta = %d" % (iter, delta))
            if delta <= delta_threshold:
                end_flag = True
            self.domain = new_domain

        environments = []
        for i in range(self.num_classes):
            index = torch.where(self.domain == i)[0]
            tempx = (self.X[index, :]).reshape(-1, self.X.shape[1])
            tempy = (self.y[index, :]).reshape(-1, 1)
            environments.append([tempx, tempy])
        return environments, self.domain



def comobine_envs(envs):
    X = []
    y = []
    for env in envs:
        X.append(env[0])
        y.append(env[1])
    X = torch.cat(X, dim=0)
    y = torch.cat(y, dim=0)
    return X.reshape(-1, X.shape[1]), y.reshape(-1,1)
