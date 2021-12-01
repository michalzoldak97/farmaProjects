import mathoperations as mat
from gradientdescent import minimize
import polynomial as poly

class PolyRegressor():

    '''
    should return: optimal_theta, inc, msq for val
    '''

    def __init__(self, x_train, x_val, y_train, y_val, degree=2, iter=1000, lr=.1, reg=.01):
        self.x_train = x_train
        self.x_val = x_val
        self.y_train = y_train
        self.y_val = y_val
        self.degree = degree
        self.lr = lr
        self.iter = iter
        self.reg = reg
        self.min_max = mat.min_max(x_train + x_val)


    def _get_poly_features(self, x, deg):
        x_n = mat.normalize_all_uni(x, (-1, 1), self.min_max)
        poly_feat = poly.PolynomialFeatures(deg)
        return poly_feat.transform(x_n)


    def _calc_theta(self, x, y):
        x_poly = self._get_poly_features(x, self.degree)
        starting_theta = mat.ones(len(x_poly[0]))
        opt_theta, c = minimize(starting_theta, x_poly, y, reg=self.reg, lr=self.lr, epochs=self.iter)
        opt_theta = [[el] for el in opt_theta]

        return opt_theta, c


    def calc_result(self):
        optimal_theta, inc = self._calc_theta(self.x_train, self.y_train)
        self.x_val = self._get_poly_features(self.x_val, self.degree)
        y_res = mat.matrix_multiply(self.x_val, optimal_theta)
        y_res = [el[0] + inc for el in y_res]
        return optimal_theta, inc, y_res


    def calc_polynomial(self):
        optimal_theta, inc, y_res = self.calc_result()
        msq = mat.calc_mean_error(self.y_val, y_res)
        return optimal_theta, inc, msq
        