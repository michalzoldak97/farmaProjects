from math import sqrt, floor


def _euc_dist(r1, r2):
    dist = .0
    for i, _ in enumerate(r1):
        dist += (r1[i] - r2[i]) ** 2
    return sqrt(dist)


def coverage_measure(y_pred, y_true):
    label = [float(i) for i in range(6)]
    shoots = [0 for _ in range(6)]
    for i, pred in enumerate(y_pred):
        diff = _euc_dist([pred], [y_true[i]])
        for res in label:
            if diff == res:
                shoots[int(res)] += 1

    cov_str = ""
    for i, val in enumerate(shoots):
        cov_str += "{}= {}  =>  {}%\n".format(label[i], val, (sum(shoots[:i+1]) / len(y_pred) * 100))

    return cov_str


def get_range(len_set: int, n_threads: int):
    step = floor(len_set / n_threads)
    val_rng = [x for x in range(0, len_set - step, step)]
    last_max = len_set % n_threads
    val_rng.append(val_rng[-1] + step)

    for i, val in enumerate(val_rng):
        up = step
        if i == len(val_rng) - 1:
            up = last_max
            if up == 0:
                up = step
        val_rng[i] = (val, val + up)

    return val_rng
