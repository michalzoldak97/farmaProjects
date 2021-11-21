import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
import fileinput
import sys
import argparse
import matchoperations as mat
import utils as util
from sklearn.preprocessing import PolynomialFeatures

mpl.rcParams['figure.figsize'] = (12, 8)


def get_filepath():
    args = argparse.ArgumentParser()
    args.add_argument("-t", "--set")
    return args.parse_args().set


def get_data(filepath):
    file = []
    for line in fileinput.input(files=filepath):
        file.append([float(i) for i in line.split()])
    x = []
    y = []
    for col in file:
        x.append(col[:-1])
        y.append(col[-1])
    return x, y


def plot_data_new(x, y, theta=np.array(([0], [0])), reg=0):
    plt.figure(figsize=(12, 8))
    plt.scatter(x, y, s=50, c='red', marker='x', linewidths=1, label='Data')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_data(x, y, theta=np.array(([0], [0])), reg=0):
    plt.figure(figsize=(12, 8))
    plt.scatter(x[:, 1], y, s=50, c='red', marker='x', linewidths=1, label='Data')
    plt.grid(True)
    plt.title('Water Data')
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    if theta.any() != 0:
        plt.plot(np.linspace(x.min(), x.max()), theta[0] + theta[1] * np.linspace(x.min(), x.max()),
                 label='Optimized linear fit')
        plt.title('Water Data: Linear Fit')

    plt.legend()


def cost(theta, X, y, reg=0):
    m = y.size
    h = np.dot(X, theta).reshape((m, 1))
    J1 = (1 / (2 * m)) * np.sum(np.square(h - y))
    J2 = (reg / (2 * m)) * theta[1:].T.dot(theta[1:])
    J = J1 + J2
    grad = ((1 / m) * (X.T.dot(h - y)) + (reg / m) * np.r_[[[0]], theta[1:].reshape(-1, 1)]).ravel()

    return J, grad


def optimal_theta(theta, X, y, reg=0):
    # Nelder-Mead yields best fit
    res = minimize(fun=cost, x0=theta, args=(X, y, reg), method='Nelder-Mead', jac=True)

    return res.x


def get_poly_features(x, deg):
    n_dim = len(x[0])
    x = util.normalize(x)
    for i in range(deg):
        for j, col in enumerate(x):
            for el in range(n_dim):
                x[j].append(col[el] ** (i + 2))
    x = [[1.0] + col for col in x]
    return x


def plot_fit(X, y, degree, num_points, reg = 0):
    x_poly = get_poly_features(X, degree)
    starting_theta = util.ones(len(x_poly[0]))
    print(starting_theta)
    sys.exit()
    opt_theta = optimal_theta(starting_theta, x_poly, y, reg)
    x_range = np.linspace(-55, 50, num_points)
    x_range_poly = np.ones((num_points, 1))
    x_range_poly = np.insert(x_range_poly, x_range_poly.shape[1], x_range.T, axis = 1)
    x_range_poly = get_poly_features(x_range_poly, len(starting_theta)-2)[1]
    y_range = x_range_poly @ opt_theta
    plot_data(X, y)
    plt.plot(x_range, y_range, "--", color = "blue", label = "Polynomial regression fit")
    plt.title('Polynomial Regression Fit: No Regularization')
    if reg != 0:
        plt.title('Polynomial Regression Fit: Lambda = {0}'.format(reg))
    plt.legend()
    plt.show()


x_train, y_train = get_data(get_filepath())
# data = loadmat('ex5data1.mat')
# y_train = data['y']
# x_train = np.c_[np.ones_like(data['X']), data['X']]
# print(y_train)
# mat.normalize_matrix(x_train)
plot_fit(x_train, y_train, 2, 10, 1)
#
# X = [[2, 3], [4, 5]]
# print(X[0])
# print(X[1])
# poly = PolynomialFeatures(3)
# print(poly.fit_transform(X))