import numpy as np
import matchoperations as mat


opt = []


def _calc_comb_num(f, deg):
    global opt
    m = len(f[0])
    num = (deg + 1) ** m
    ls = [0.0] * m

    for x in range(1, num):
        for idx in range(len(ls) - 1, -1, -1):
            if ls[idx] == 0:
                ls[idx] += 1
                break
            elif ls[idx] % deg == 0:
                ls[idx] = 0
                if ls[idx - 1] != deg:
                    ls[idx - 1] += 1
                    break
                else:
                    dfc = -1
                    stop = False
                    while not stop and (idx + dfc) > -1:
                        if ls[idx + dfc] != deg:
                            ls[idx + dfc] += 1
                            stop = True
                        else:
                            ls[idx + dfc] = 0
                            dfc -= 1
                    break
            elif idx == (len(ls) - 1):
                ls[idx] += 1
                break
    
        if sum(ls) == deg:
            opt.append(tuple(ls))


def _calc_poly_features(f, deg):
    global opt
    _calc_comb_num(f, deg)
    poly_features = np.empty((len(f),0))
    feature_ls = []

    for i in range(f.shape[1]):
        feature_ls.append(f[:, i].reshape(-1, 1))

    for i, tup in enumerate(opt):
        vec = np.full((len(f), 1), 1.0)

        for f_idx, exp in enumerate(tup):
            vec *= feature_ls[f_idx] ** exp

        poly_features = np.c_[poly_features, vec]

    return poly_features


def get_poly_features(f, deg, min_max):
    f = mat.normalize_all_uni(f, (-1, 1), min_max)
    f = np.asanyarray(f)
    tran = np.ones((len(f), 1))

    for i in range(1, deg + 1):
        feat_for_deg = _calc_poly_features(f, i)
        tran = np.c_[tran, feat_for_deg]
    
    return np.ndarray.tolist(tran)

class PolynomialFeaturesForDegree():
    def __init__(self, degree):
        self.degree = degree
        self.options = []

    
    def get_terms(self, features):
        m = features.shape[1]
        num = (self.degree + 1) ** m
        ls = [0] * m

        for x in range(1, num):
            for idx in range(len(ls) - 1, -1, -1):
                if ls[idx] == 0:
                    ls[idx] += 1
                    break
                
