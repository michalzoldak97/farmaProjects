import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import mathoperations as mat
import utils as util
from gradientdescent import minimize

mpl.rcParams['figure.figsize'] = (12, 8)


def plot_data_new(x, y, theta=np.array(([0], [0])), reg=0):
    plt.figure(figsize=(12, 8))
    plt.scatter(x, y, s=50, c='red', marker='x', linewidths=1, label='Data')
    plt.grid(True)
    plt.legend()


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


def get_poly_features(x, deg):
    n_dim = len(x[0])
    x = mat.normalize(x)
    for i in range(deg):
        for j, col in enumerate(x):
            for el in range(n_dim):
                x[j].append(col[el] ** (i + 2))
    x = [[1.0] + col for col in x]
    return x


def plot_fit(X, y, degree, num_points, reg = 0):
    x_poly = get_poly_features(X, degree)
    starting_theta = mat.ones(len(x_poly[0]))
    opt_theta = minimize(starting_theta, x_poly, y, reg=0)
    opt_theta = [[el] for el in opt_theta]
    x_range = np.linspace(-55, 50, num_points)
    x_range_poly = [[el] for el in x_range]
    x_range_poly = get_poly_features(x_range_poly, len(starting_theta)-2)
    y_range = mat.matrix_multiply(x_range_poly, opt_theta)
    print(y_range)
    plot_data_new(X, y)
    plt.plot(x_range, y_range, "--", color = "blue", label = "Polynomial regression fit")
    plt.title('Polynomial Regression Fit: No Regularization')
    if reg != 0:
        plt.title('Polynomial Regression Fit: Lambda = {0}'.format(reg))
    plt.legend()
    plt.show()


x_train, y_train = util.get_data(util.get_filepath())
# data = loadmat('ex5data1.mat')
# y_train = data['y']
# x_train = np.c_[np.ones_like(data['X']), data['X']]
# print(y_train)
# mat.normalize_matrix(x_train)
plot_fit(x_train, y_train, 4, 100, 0)
#
# X = [[2, 3], [4, 5]]
# print(X[0])
# print(X[1])
# poly = PolynomialFeatures(3)
# print(poly.fit_transform(X))