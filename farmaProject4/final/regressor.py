from polyoperations import PolyRegressor
import utils as util
import copy


def train(reglr: list):
    global min_deg, degree_range, x_tr, y_tr, x_vl, y_vl, res_tpl
    for rg in reglr:
        test_res = [0, 0.0]
        for deg in degree_range:
            a, b, c, d = copy.deepcopy(x_tr), copy.deepcopy(y_tr), copy.deepcopy(x_vl), copy.deepcopy(y_vl)
            poly_reg = PolyRegressor(a, c, b, d, degree=deg, iter_=1000, reg=rg)
            res = poly_reg.calc_polynomial()[2]
            if deg > min_deg and res < test_res[1]:
                test_res = [deg, res]
            elif deg == min_deg:
                test_res = [deg, res]
        if test_res[1] < res_tpl[1]:
            res_tpl[0], res_tpl[1] = test_res[0], test_res[1]
            res_tpl[2] = rg


x_train, y_train = util.get_data(util.get_filepath())
x_test = util.read_sys_file()

x_tr, y_tr, x_vl, y_vl = util.train_test_split(x_train, y_train, .7)
res_tpl = [0, 999.0, 0.0]

min_deg = 1
degree_range = range(min_deg, 12)

train([.0, .1, .2, .3, .4, .5])

final_poly_reg = PolyRegressor(x_train, x_test, y_train, y_train, degree=res_tpl[0], iter_=2000, reg=res_tpl[2])
final_res = final_poly_reg.calc_result()[2]

for r in final_res:
    print(r)
