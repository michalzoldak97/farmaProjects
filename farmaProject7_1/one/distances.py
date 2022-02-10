from math import sqrt


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

    first = (sum(shoots[:1]) / len(y_pred) * 100)
    second = (sum(shoots[:2]) / len(y_pred) * 100)

    return cov_str, first, second
