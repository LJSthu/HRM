import numpy as np
from numpy.random import seed
seed(1)
import torch
import math
import random
random.seed(1)

def sign(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0

def data_generation(n1, n2, ps, pvb, pv, r):
    S = np.random.normal(0, 1, [n1, ps])
    V = np.random.normal(0, 1, [n1, pvb + pv])

    Z = np.random.normal(0, 1, [n1, ps + 1])
    for i in range(ps):
        S[:, i:i + 1] = 0.8 * Z[:, i:i + 1] + 0.2 * Z[:, i + 1:i + 2]

    beta = np.zeros((ps, 1))
    for i in range(ps):
        beta[i] = (-1) ** i * (i % 3 + 1) * 1.0 / 2

    noise = np.random.normal(0, 0.3, [n1, 1])

    Y = np.dot(S, beta) + noise + 1 * S[:, 0:1] * S[:, 1:2] * S[:, 2:3]
    Y_compare = np.dot(S, beta) + 1 * S[:, 0:1] * S[:, 1:2] * S[:, 2:3]

    index_pre = np.ones([n1, 1], dtype=bool)
    for i in range(pvb):
        D = np.abs(V[:, pv + i:pv + i + 1] * sign(r) - Y_compare)
        pro = np.power(np.abs(r), -D * 5)
        selection_bias = np.random.random([n1, 1])
        index_pre = index_pre & (
                    selection_bias < pro)
    index = np.where(index_pre == True)
    S_re = S[index[0], :]
    V_re = V[index[0], :]
    Y_re = Y[index[0]]
    n, p = S_re.shape
    index_s = np.random.permutation(n)

    X_re = np.hstack((S_re, V_re))
    beta_X = np.vstack((beta, np.zeros((pv + pvb, 1))))

    return torch.Tensor(X_re[index_s[0:n2], :]), torch.Tensor(Y_re[index_s[0:n2], :]), beta_X


def modified_selection_bias(ps, pv, n, r):
    S = np.random.normal(0, 1, [n, ps])
    Z = np.random.normal(0, 1, [n, ps + 1])
    for i in range(ps):
        S[:, i:i + 1] = 0.8 * Z[:, i:i + 1] + 0.2 * Z[:, i + 1:i + 2]

    beta = np.zeros((ps, 1))
    for i in range(ps):
        beta[i] = (-1) ** i * (i % 3 + 1) * 1.0 / 3

    noise = np.random.normal(0, 0.3, [n, 1])

    Y = np.dot(S, beta) + noise + 1 * S[:, 0:1] * S[:, 1:2] * S[:, 2:3]
    Y_compare = np.dot(S, beta) + 1 * S[:, 0:1] * S[:, 1:2] * S[:, 2:3]

    if r > 0:
        center = Y_compare
    else:
        center = -Y_compare

    r = abs(r)
    sigma = math.sqrt(1/math.log2(r))

    V = np.zeros((center.shape[0], pv), dtype=np.float32)
    for i in range(center.shape[0]):
        V[i,:] = np.random.multivariate_normal(center[i]*(np.zeros(pv)+1.0), sigma*np.eye(pv), 1)

    X = np.concatenate((S,V), axis=1)
    X = torch.Tensor(X)
    Y = torch.Tensor(Y)
    return X, Y

def modified_Multi_env_selection_bias():
    trainX = None
    trainy = None
    env = []
    n_list = [1900, 100, 100]
    r_list = [1.9, -1.1, -1.1]
    ps = 5
    pv = 5
    for e in range(len(n_list)):
        if trainy is None:
            trainX, trainy = modified_selection_bias(ps, pv, n_list[e], r_list[e])
            env.append([trainX, trainy])
        else:
            tempx, tempy = modified_selection_bias(ps, pv, n_list[e], r_list[e])
            trainX = np.concatenate([trainX, tempx], axis=0)
            trainy = np.concatenate([trainy, tempy], axis=0)
            env.append([tempx, tempy])

    return env, 0



def Multi_env_selection_bias():
    n1 = 100000
    p = 10
    ps = int(p * 0.5)
    pvb = int(p * 0.1)
    pv = p - ps - pvb

    r = 1.5
    r_list = [-1.1]
    num_list = [100]
    environments = []
    n2 = 1900
    trainx, trainy, real_beta = data_generation(n1, n2, ps, pvb, pv, r)
    environments.append([trainx, trainy])

    for idx, r in enumerate(r_list):
        x_bias, y_bias, real_beta = data_generation(n1, num_list[idx], ps, pvb, pv, r)
        environments.append([x_bias, y_bias])
    print(environments[0][0].shape, environments[0][1].shape, environments[1][0].shape)
    return environments, real_beta



def generate_test():
    n1 = 100000
    p = 10
    ps = int(p * 0.5)
    pvb = int(p * 0.1)
    pv = p - ps - pvb

    r_list = [-3, -2.7, -2.3, -2.0, -1.7,1.7,2.0,2.3,2.7,3.0]
    testing = []

    for r in r_list:
        n2 = 2000
        trainx, trainy, real_beta = data_generation(n1, n2, ps, pvb, pv, r)
        testing.append([trainx, trainy])

    return testing

if __name__=="__main__":
    Multi_env_selection_bias()