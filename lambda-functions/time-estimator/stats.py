import math
import random


def safe_log(x):
    return math.log(max(x, 10 ** -100))


def safe_sigmoid(x):
    return 1. / (1. + math.exp(min(x, 750)))


def normal_pdf(x, mu, sigma):
    return math.exp(-((x - mu) ** 2) / (2 * (sigma ** 2))) / math.sqrt(2 * math.pi * (sigma ** 2))


def normal_cdf(x, mu, sigma):
    return (math.erf((x - mu) / (math.sqrt(2) * sigma)) + 1) / 2


def single_regression(x, y):
    n = len(x)
    x_sum = sum(x)
    y_sum = sum(y)
    xy_sum = sum(x * y for x, y in zip(x, y))
    sqx_sum = sum(x ** 2 for x in x)
    slope = (n * xy_sum - x_sum * y_sum) / (n * sqx_sum - x_sum ** 2)
    intercept = (sqx_sum * y_sum - xy_sum * x_sum) / (n * sqx_sum - x_sum ** 2)
    return slope, intercept


def minimize(args, samples, f, grad_f, max_iter, eta=1., rand_state=20191019):
    args = list(args)
    samples = list(samples)
    dim = len(args)

    random.seed(rand_state)

    iter_n = max(max_iter // len(samples), 1)

    rs = [0.] * dim
    iterations = []
    for iteration in range(iter_n):
        score = sum(f(sample, args) for sample in samples)
        iterations.append((score, list(args)))

        random.shuffle(samples)
        for sample in samples:
            grads = grad_f(sample, args)
            for i in range(dim):
                if grads[i] == 0.:
                    continue
                rs[i] += grads[i] ** 2
                args[i] -= eta * grads[i] / rs[i] ** 0.5

    best_score, arg = min(iterations)
    return arg, best_score
