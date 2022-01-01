from math import sqrt


def calc_gini_idx(gr, cls):
    n_inst = float(sum([len(g) for g in gr]))
    idx = .0
    for g in gr:
        size = float(len(g))
        if size > 0:
            score = .0
            for class_val in cls:
                p = [row[-1] for row in g].count(class_val) / size
                score += p * p

            idx += (1. - score) * (size / n_inst)

    return idx


def euc_dist(r1, r2):
    dist = 0.0
    for i, _ in enumerate(r1):
        dist += (r1[i] - r2[i]) ** 2
    return sqrt(dist)


def coverage_measure(y_pred, y_true):
    label = [float(i) for i in range(6)]
    shoots = [0 for _ in range(6)]
    for i, pred in enumerate(y_pred):
        diff = euc_dist([pred], [y_true[i]])
        for res in label:
            if diff == res:
                shoots[int(res)] += 1

    cov_str = ""
    for i, val in enumerate(shoots):
        cov_str += "{}= {}  =>  {}%\n".format(label[i], val, (sum(shoots[:i+1]) / len(y_pred) * 100))

    return cov_str
