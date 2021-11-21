import matchoperations as mat
import utils as util
from gradientdescent import minimize


def _get_poly_features(x, deg):
    n_dim = len(x[0])
    x = mat.normalize(x)
    for i in range(deg):
        for j, col in enumerate(x):
            for el in range(n_dim):
                x[j].append(col[el] ** (i + 2))
    x = [[1.0] + col for col in x]

    return x


def _calc_theta(x, y, degree, reg=0):
    x_poly = _get_poly_features(x, degree)
    starting_theta = mat.ones(len(x_poly[0]))
    opt_theta = minimize(starting_theta, x_poly, y, reg=0)
    opt_theta = [[el] for el in opt_theta]

    return opt_theta


x_train, y_train = util.get_data(util.get_filepath())
opt = _calc_theta(x_train, y_train, len(x_train[0]) + 2, 0)
x_test = _get_poly_features(util.read_sys_file(), len(opt) - 2)
y_res = mat.matrix_multiply(x_test, opt)

y_res = [x[0] for x in y_res]
for r in y_res:
    print(r)
