from Backend import MpModel, pretty
from Frontend import McModel
from Selection_bias import Multi_env_selection_bias, generate_test, modified_Multi_env_selection_bias
import torch
import numpy as np
import os
from sklearn.linear_model import LinearRegression



class HRM:
    def __init__(self, front_params, back_params, X, y, test_X=None, test_y=None):
        self.X = X
        self.y = y
        self.test_X = [test_X]
        self.test_y = [test_y]
        self.frontend = McModel(front_params['num_clusters'], self.X, self.y)
        self.backend = MpModel(input_dim=back_params['input_dim'],
                                    output_dim=back_params['output_dim'],
                                    sigma=back_params['sigma'],
                                    lam=back_params['lam'],
                                    alpha=back_params['alpha'],
                                    hard_sum=back_params['hard_sum'])
        self.domains = None
        self.weight = torch.tensor(np.zeros(self.X.shape[1], dtype=np.float32))


    def solve(self, iters):
        self.density_result = None
        density_record = []
        flag = False
        for i in range(iters):
            environments, self.domains = self.frontend.cluster(self.weight, self.domains, flag)
            weight, density = self.backend.train(environments, epochs=6000)
            density_record.append(density)
            self.density_result = density
            self.weight = density
            self.backend.lam *= 1.05
            self.backend.alpha *= 1.05
            print('Selection Ratio is %s' % self.weight)

        f = open('./save.txt', 'a+')
        print('Density results:')
        for i in range(len(density_record)):
            print("Iter %d Density %s" % (i, pretty(density_record[i])))
            f.writelines(pretty(density_record[i]) + '\n')
        f.close()
        return self.weight

    def test(self, test_envs):
        test_accs = []
        self.backend.backmodel.eval()
        self.backend.featureSelector.eval()
        for i in range(len(test_envs)):
            pred = self.backend.single_forward(test_envs[i][0])
            error = torch.sqrt(torch.mean((pred.reshape(test_envs[i][1].shape) - test_envs[i][1]) ** 2))
            test_accs.append(error.data)

        print(pretty(test_accs))
        self.backend.backmodel.train()
        self.backend.featureSelector.train()
        return test_accs


def combine_envs(envs):
    X = []
    y = []
    for env in envs:
        X.append(env[0])
        y.append(env[1])
    X = torch.cat(X, dim=0)
    y = torch.cat(y, dim=0)
    return X.reshape(-1, X.shape[1]), y.reshape(-1, 1)


def seed_torch(seed=2018):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class EmpiricalRiskMinimizer(object):
    def __init__(self, X, y, mask):
        x_all = X.numpy()
        y_all = y.numpy()
        self.mask = mask
        x_all = x_all[:, self.mask]
        w = LinearRegression(fit_intercept=False).fit(x_all, y_all).coef_
        self.w = torch.Tensor(w)

    def solution(self):
        return self.w

    def test(self, X, y):
        X = X.numpy()
        X = X[:, self.mask]
        y = y.numpy()
        err = np.mean((X.dot(self.w.T) - y) ** 2.).item()
        return np.sqrt(err)


if __name__ == "__main__":
    all_weights = torch.tensor(np.zeros(10, dtype=np.float32))
    average = 0.0
    std = 0.0
    seeds = 10
    average_error_list = torch.Tensor(np.zeros(10, dtype=np.float))
    for seed in range(0, seeds):
        seed_torch(seed)
        print("---------------seed = %d------------------" % seed)
        environments, _ = Multi_env_selection_bias()
        X, y = combine_envs(environments)

        # params
        front_params = {}
        front_params['num_clusters'] = 3

        back_params = {}
        back_params['input_dim'] = X.shape[1]
        back_params['output_dim'] = 1
        back_params['sigma'] = 0.1
        back_params['lam'] = 0.1
        back_params['alpha'] = 1000.0
        back_params['hard_sum'] = 10
        back_params['overall_threshold'] = 0.20
        whole_iters = 5

        # train and test
        model = HRM(front_params, back_params, X, y)
        result_weight = model.solve(whole_iters)
        all_weights += result_weight

        mask = torch.where(result_weight > back_params['overall_threshold'])[0]
        evaluate_model = EmpiricalRiskMinimizer(X, y, mask)
        testing_envs = generate_test()

        testing_errors = []
        for [X, y] in testing_envs:
            testing_errors.append(evaluate_model.test(X, y))

        testing_errors = torch.Tensor(testing_errors)
        print(testing_errors)
        average += torch.mean(testing_errors) / seeds
        std += torch.std(testing_errors) / seeds
        average_error_list += testing_errors / seeds
        print(average_error_list)

    print(average)
    print(std)
    print(all_weights)






