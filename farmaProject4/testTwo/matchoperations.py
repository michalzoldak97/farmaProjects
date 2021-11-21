import math


def _get_matrix_cell(m, col):
    return [m[i][col]for i, _ in enumerate(m)]


def _calc_matrix_cell(ma, mb, i, j):
    mb_col = _get_matrix_cell(mb, j)
    prod = [ma[i][x] * mb_col[x] for x, _ in enumerate(mb_col)]

    return sum(prod)


def matrix_multiply(ma, mb):
    a_dim = (len(ma), len(ma[0]))
    b_dim = (len(mb), len(mb[0]))
    res = [[0.0 for p in range(b_dim[1])] for q in range(a_dim[0])]
    for i in range(a_dim[0]):
        for j in range(b_dim[1]):
            res[i][j] = _calc_matrix_cell(ma, mb, i, j)

    return res


def inverse(x):
    x_inv = [[] for i in x[0]]
    for col in x:
        for i, el in enumerate(col):
            x_inv[i].append(el)

    return x_inv


def normalize(x):
    x_rev = inverse(x)
    for col in x_rev:
        x_max = max(col)
        x_min = min(col)
        for i, x_pr in enumerate(col):
            try:
                col[i] = (x_pr - x_min) / (x_max - x_min)
            except ZeroDivisionError:
                continue

    return inverse(x_rev)


def normalize_universal(x, ab):
    x_rev = inverse(x)
    for col in x_rev:
        x_max = max(col)
        x_min = min(col)
        for i, x_pr in enumerate(col):
            try:
                col[i] = ab[0] + ((x_pr - x_min)*(ab[1] - ab[0])) / (x_max - x_min)
            except ZeroDivisionError:
                continue

    return inverse(x_rev)


def _mean(x):
    return sum(x) / len(x)


def _std(x):
    mean = _mean(x)
    return mean, math.sqrt(sum([(a - mean) ** 2 for a in x]) / len(x))


def normalize_features(x):
    x_rev = inverse(x)
    for col in x_rev:
        x_mean, x_std = _std(col)
        if x_std != 0.0:
            for i, x_pr in enumerate(col):
                col[i] = (x_pr - x_mean) / x_std

    return inverse(x_rev)


def ones(l):
    return [1.0 for _ in range(l)]
