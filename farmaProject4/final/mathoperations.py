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


def normalize_all_uni(x, ab, min_max):
    for col in x:
        for i, x_pr in enumerate(col):
            col[i] = ab[0] + ((x_pr - min_max[0]) * (ab[1] - ab[0])) / (min_max[1] - min_max[0])

    return x


def min_max(x_all):
    x_min = min(x_all[0])
    x_max = max(x_all[0])
    for col in x_all:
        col_min = min(col)
        col_max = max(col)
        if col_min < x_min:
            x_min = col_min
        if col_max > x_max:
            x_max = col_max
    
    return x_min, x_max


def _mean(x):
    return sum(x) / len(x)


def _std(x):
    mean = _mean(x)
    return mean, math.sqrt(sum([(a - mean) ** 2 for a in x]) / len(x))


def calc_mean_error(y, y_pred):
    return (1 / len(y)) * sum([(a - y_pred[i]) ** 2 for i, a in enumerate(y)])


def ones(l):
    return [1.0 for _ in range(l)]
