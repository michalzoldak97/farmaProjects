import matchoperations as mat
import utils as util
from gradientdescent import minimize
from polyfeatures import get_poly_features
import sys


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
    global min_max
    # x_poly = get_poly_features(x, degree, min_max)
    # print(x_poly)
    starting_theta = mat.ones(len(x_poly[0]))
    opt_theta, c = minimize(starting_theta, x_poly, y, reg=reg)
    opt_theta = [[el] for el in opt_theta]

    return opt_theta, c


x_train, y_train = util.get_data(util.get_filepath())
x_test = util.read_sys_file()
# print(x_train)
min_max = mat.min_max(x_train + x_test)
opt, inter = _calc_theta(x_train, y_train, 6, 0.025)
# bprint("Opt: {}".format(opt))
x_test = _get_poly_features(x_test, 6)
#x_test = get_poly_features(x_test, 3, min_max)
# print("X test: {}".format(x_test))
y_res = mat.matrix_multiply(x_test, opt)
y_res = [el[0] + inter for el in y_res]
y_test = open("res_5d", "r")
y_test = [float(col.split()[-1]) for col in y_test]

print(mat.calc_mean_error(y_test, y_res))

for r in y_res:
    print(r)
