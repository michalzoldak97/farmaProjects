from polyoperations import PolyRegressor
import utils as util
import copy


x_train, y_train = util.get_data(util.get_filepath())
x_test = util.read_sys_file()

x_tr, y_tr, x_vl, y_vl = util.train_test_split(x_train, y_train, .75)
res_tpl = [0, 0.0]

min_degree, max_degree = 1, 12

for deg in range(min_degree, max_degree + 1):
    a, b, c, d = copy.deepcopy(x_tr), copy.deepcopy(y_tr), copy.deepcopy(x_vl), copy.deepcopy(y_vl)
    poly_reg = PolyRegressor(a, c, b, d, degree=deg, iter=5000, reg=.1)
    res = poly_reg.calc_polynomial()[2]
    if deg > min_degree and res < res_tpl[1]:
        res_tpl = [deg, res]
    elif deg == min_degree:
        res_tpl = [deg, res]


final_poly_reg = PolyRegressor(x_train, x_test, y_train, y_train, degree=res_tpl[0], iter=5000, reg=.1)
final_res = final_poly_reg.calc_result()[2]

for r in final_res:
    print(r)
