import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
import gradientdescent as grad
import mathoperations as mat
from sklearn.preprocessing import PolynomialFeatures
import fileinput
import sys
import argparse

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
    res_desc = grad.minimize(theta, X, y, reg)
    return res_desc # res.x


def poly_features(X, p):
    for i in np.arange(p):
        dim = i + 2
        X = np.insert(X, X.shape[1], np.power(X[:, 1], dim), axis=1)

    X_norm = X
    # column wise
    means = np.mean(X_norm, axis=0)
    X_norm[:, 1:] = X_norm[:, 1:] - means[1:]
    stds = np.std(X_norm, axis=0)
    X_norm[:, 1:] = X_norm[:, 1:] / stds[1:]
    return X, X_norm


def plot_fit(X, y, degree, num_points, reg = 0):
    X_poly = poly_features(X, degree)[1]
    starting_theta = np.ones((X_poly.shape[1], 1))
    opt_theta = optimal_theta(starting_theta, X_poly, y, reg)
    # x_range = np.linspace(-45, 40, num_points)
    x_range = np.array([-48.058829452570066, -44.38375985168692, -34.70626581132249, -29.152979217238133,
                        -15.93675813378541, -8.941457938049755, 1.3891543686358903, 7.013502082404112,
                        15.307792889226079, 22.762748919711303, 36.18954862666253, 37.49218733199513])
    x_range_poly = np.ones((num_points, 1))
    x_range_poly = np.insert(x_range_poly, x_range_poly.shape[1], x_range.T, axis = 1)
    x_range_poly = poly_features(x_range_poly, len(starting_theta)-2)[0]
    y_range = x_range_poly @ opt_theta
    plot_data(X, y)
    plt.plot(x_range, y_range, "--", color = "blue", label = "Polynomial regression fit")
    plt.title('Polynomial Regression Fit: No Regularization')
    if reg != 0:
        plt.title('Polynomial Regression Fit: Lambda = {0}'.format(reg))
    plt.legend()
    plt.show()
    return y_range


x_train, y_train = get_data(get_filepath())
y_train_p = y_train
x_train_p = x_train
x_train = np.c_[np.ones_like(x_train), x_train]
x_train = np.array(x_train)
y_train = np.array(y_train)
# data = loadmat('ex5data1.mat')
# y_train = data['y']
# x_train = np.c_[np.ones_like(data['X']), data['X']]
y_pred = plot_fit(x_train, y_train, 8, 12, 0)

##///##\\\

x_train_p = [x[0] for x in x_train_p]
y_pred = [x[0] for x in y_pred]
plt.scatter(x_train_p, y_train_p, s=50, c='red', marker='x', linewidths=1, label='Data')
plt.plot(x_train_p, y_pred, "--", color="blue", label="Polynomial regression fit")
plt.show()
# print(y_train_p)
# print(y_pred)
print(mat.calc_mean_error(y_train_p, y_pred))
