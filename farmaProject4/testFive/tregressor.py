import matchoperations as mat
import utils as util
from gradientdescent import minimize
import matplotlib.pyplot as plt


def _plot_result(x, y):
    plt.scatter(x, y, s=50, c='red', marker='x', linewidths=1, label='Data')
    plt.grid(True)


def _get_poly_features(x, deg):
    global min_max
    n_dim = len(x[0])
    x = mat.normalize_all_uni(x, (-1, 1), min_max)
    for i in range(deg):
        for j, col in enumerate(x):
            for el in range(n_dim):
                x[j].append(col[el] ** (i + 2))
    x = [[1.0] + col for col in x]

    return x


def _calc_theta(x, y, degree, reg=0.0):
    x_poly = _get_poly_features(x, degree)
    starting_theta = mat.ones(len(x_poly[0]))
    opt_theta = minimize(starting_theta, x_poly, y, reg=reg)
    opt_theta = [[el] for el in opt_theta]

    return opt_theta


x_train, y_train = util.get_data(util.get_filepath())
x_test = util.read_sys_file()
x_test_to_prt = [el[0] for el in x_test]
min_max = mat.min_max(x_train + x_test)
opt = _calc_theta(x_train, y_train, len(x_train[0])*6, 0.01)
x_test = _get_poly_features(x_test, len(opt) - 2)
print(len(x_test[0]))
print(len(opt[0]))
y_res = mat.matrix_multiply(x_test, opt)

_plot_result(x_test_to_prt, y_res)
plt.show()

for r in y_res:
    print(r[0])
