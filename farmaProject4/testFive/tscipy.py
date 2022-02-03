import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
import matchoperations as mat
import utils as util
from gradientdescent import minimize
import sys


def _plot_result(x, y):
    plt.scatter(x, y, s=50, c='red', marker='x', linewidths=1, label='Data')
    plt.grid(True)


x_train, y_train = util.get_data(util.get_filepath())
x_test = util.read_sys_file()
x_range = x_test
#print(x_train)
poly = PolynomialFeatures(8)
x_train_poly = poly.fit_transform(x_train)
x_train_poly = [[el for el in col] for col in x_train_poly]
linear_model = LinearRegression()
linear_model.fit(x_train_poly, y_train)
ridge = Ridge()
ridge.fit(x_train_poly, y_train)
x_test = poly.fit_transform(x_test)
coeffs = [[el] for el in linear_model.coef_]
ridge_coeffs = [[el] for el in ridge.coef_]
x_test = [[el for el in col] for col in x_test]
y_test_lr = linear_model.intercept_ + mat.matrix_multiply(x_test, coeffs)
print("intercept", linear_model.intercept_)
print("multi", mat.matrix_multiply(x_test, coeffs))
print("y test", y_test_lr)
y_test_ridge = ridge.intercept_ + mat.matrix_multiply(x_test, ridge_coeffs)
y_test_lr = [el[0] for el in y_test_lr]
y_test_ridge = [el[0] for el in y_test_ridge]
y_test = open("res_5d", "r")
y_test = [float(col.split()[-1]) for col in y_test]
print(mat.calc_mean_error(y_test, y_test_lr))
print(mat.calc_mean_error(y_test, y_test_ridge))

for y in y_test_lr:
    print(y)