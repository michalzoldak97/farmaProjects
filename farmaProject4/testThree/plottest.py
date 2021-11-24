import random
import matplotlib.pyplot as plt
import numpy as np
import math


def func_1d(x):
    return x ** 3 + (2 * x) ** 2 - (0.25 * x) ** 4 + 3 + 100 * math.cos(x) ** 2


def func_5d(x):
    return 5 + x[0] ** 2 + x[1] - x[2] ** 2 + 17 * x[3] - 4 * x[4]


def plot_func_1d(x, y):
    plt.plot(x, y, "--", color="blue", label="Polynomial")
    plt.show()


def plot_points_1d(x, y):
    plt.scatter(x, y, s=20, c='red', marker='x', linewidths=1, label='Data')
    plt.show()


def write_all1d(x, y):
    all_1d = open("all_1d", "w")
    for i, xarg in enumerate(x):
        all_1d.write(str(xarg) + " " + str(y[i]) + "\n")
    all_1d.close()


def write_sample1d(x, y, m):
    samples = random.sample(range(len(y)), m)
    sampl_1d = open("samples_1d", "w")
    for i in samples:
        sampl_1d.write(str(x[i]) + " " + str(y[i]) + "\n")
    sampl_1d.close()

    in_1d = open("in_1d", "w")
    res_1d = open("res_1d", "w")
    for i, xarg in enumerate(x):
        if i not in samples:
            in_1d.write(str(xarg) + "\n")
            res_1d.write(str(y[i]) + "\n")
    in_1d.close()
    res_1d.close()


def create_func_1d():
    x_range = np.linspace(-10, 10, 200)
    x_range = [el for el in x_range]
    y_range = [func_1d(x) for x in x_range]
    y_range = [x + 0.1 * (random.uniform(-np.std(y_range), np.std(y_range))) for x in y_range]
    write_all1d(x_range, y_range)
    write_sample1d(x_range, y_range, 160)

    plot_points_1d(x_range, y_range)


def create_func_5d():
    x_range = [np.linspace(-10, 10, 10) for _ in range(5)]
    x_range = [[el for el in x_dim] for x_dim in x_range]
    y_range = []
    for i, _ in enumerate(x_range[0]):
        vec = []
        for j, _ in enumerate(x_range):
            vec.append(x_range[j][i])
        y_range.append(func_5d(vec))
    y_range = [x + 0.1 * (random.uniform(-np.std(y_range), np.std(y_range))) for x in y_range]


create_func_5d()
