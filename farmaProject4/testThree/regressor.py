import mathoperations as mat
import utils as util
from gradientdescent import minimize
import matplotlib.pyplot as plt


def _plot_result(x, y):
    plt.scatter(x, y, s=50, c='red', marker='x', linewidths=1, label='Data')
    plt.grid(True)


def _get_poly_features(x, deg):
    n_dim = len(x[0])
    x = mat.normalize_universal(x, (-1, 1))
    for i in range(deg):
        for j, col in enumerate(x):
            for el in range(n_dim):
                x[j].append(col[el] ** (i + 2))
    x = [[1.0] + col for col in x]

    return x


def _calc_theta(x, y, degree, reg):
    x_poly = _get_poly_features(x, degree)
    starting_theta = mat.ones(len(x_poly[0]))
    opt_theta = minimize(starting_theta, x_poly, y, reg=reg)
    opt_theta = [[el] for el in opt_theta]

    return opt_theta


x_train, y_train = util.get_data(util.get_filepath())
opt = _calc_theta(x_train, y_train, 24, 0.25)
sys_x = util.read_sys_file()
x_res = _get_poly_features(sys_x, len(opt) - 2)
y_res = mat.matrix_multiply(x_res, opt)

y_res = [x[0] for x in y_res]

test_file = open("res_1d", "r")
y_test = [[float(el) for el in col.split()]for col in test_file]
y_test = [x[0] for x in y_test]
test_file.close()
_plot_result(sys_x, y_test)
plt.plot(sys_x, y_res, "--", color="blue", label="Polynomial regression fit")
plt.show()
print(mat.calc_mean_error(y_test, y_res))
for el in y_res:
    print(el)
