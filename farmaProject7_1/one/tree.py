from math import log

num_class = 6  # 0 to 5


def _calc_cls_counts(data: list):
    count = [0 for _ in range(num_class)]
    for row in data:
        count[int(row[-1])] += 1
    return count


def _calc_gini_idx(data: list):
    d_len = float(len(data))
    cls_counts = _calc_cls_counts(data)
    gini_idx = 1.
    for cls_count in cls_counts:
        plx = cls_count / d_len
        gini_idx -= plx**2
    return gini_idx


def _calc_entropy(data: list):
    d_len = float(len(data))
    cls_counts = _calc_cls_counts(data)
    ent = 0.
    for cls_count in cls_counts:
        plx = cls_count / d_len
        try:
            ent += plx * log(plx, 10)
        except ValueError:
            continue
    return ent


def _calc_dis_measure(data: list, method='gini'):
    if method == 'gini':
        return _calc_gini_idx(data)
    elif method == 'entropy':
        return _calc_entropy(data)
    else:
        raise ValueError('Invalid method')

