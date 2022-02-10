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
        gini_idx -= plx ** 2
    return gini_idx


def _calc_entropy(data: list):
    d_len = float(len(data))
    cls_counts = _calc_cls_counts(data)
    ent = 0.
    for cls_count in cls_counts:
        plx = cls_count / d_len
        if plx != 0.:
            ent += plx * log(plx, 10)
    return -ent


def _calc_dis_measure(data: list, method='gini'):
    if method == 'gini':
        return _calc_gini_idx(data)
    elif method == 'entropy':
        return _calc_entropy(data)
    else:
        raise ValueError('Invalid method')


def _calc_info_gain(dis_measure_r: float, l_set: list, r_set: list, method):
    # pl = len l / len l + len r ; pr = 1 - pl
    len_l = len(l_set)
    p = float(len_l / (len_l + len(r_set)))
    return dis_measure_r - (p * _calc_dis_measure(l_set, method) + (1. - p) * _calc_dis_measure(r_set, method))


class Condition:
    def __init__(self, col_idx: int, val: float):
        self.col_idx = col_idx
        self.val = val

    def __repr__(self):
        return "Is {} < {}".format(self.col_idx, self.val)

    def is_con(self, row: list):
        return row[self.col_idx] < self.val


def _get_lr_sets(data: list, con: Condition):
    l_set, r_set = [], []
    for row in data:
        if con.is_con(row):
            l_set.append(row)
        else:
            r_set.append(row)
    return l_set, r_set


def _get_best_split(data: list, method):
    feats_idxs = range(len(data[0]) - 1)  # range of features in vec, excluding rate col
    best_info_gain, best_con = 0, None
    start_dis_measure = _calc_dis_measure(data, method)  # Qr
    for col_idx in feats_idxs:  # foreach column
        col_vals = set([row[col_idx] for row in data])  # foreach unique value in column
        for val in col_vals:
            curr_con = Condition(col_idx, val)
            l_set, r_set = _get_lr_sets(data, curr_con)

            if len(l_set) == 0 or len(r_set) == 0:
                continue

            curr_info_gain = _calc_info_gain(start_dis_measure, l_set, r_set, method)
            if curr_info_gain > best_info_gain:
                best_info_gain, best_con = curr_info_gain, curr_con

    return best_info_gain, best_con


def _get_most_common_rate(rates: list):
    rate_nums = {}
    for rate in rates:
        if rate in rate_nums.keys():
            rate_nums[rate] += 1
        else:
            rate_nums[rate] = 1

    return max(rate_nums, key=rate_nums.get)


class Leaf:
    def __init__(self, data: list):
        rates = [row[-1] for row in data]
        self.rate = _get_most_common_rate(rates)


class Node:
    def __init__(self, con: Condition, l_branch, r_branch):
        self.condition = con
        self.l_branch = l_branch
        self.r_branch = r_branch


def build_tree(data: list, depth: int, max_depth: int, min_set_size: int, method='gini'):
    info_gain, condition = _get_best_split(data, method)

    if info_gain == 0 or depth > max_depth or len(data) < min_set_size:
        return Leaf(data)

    l_set, r_set = _get_lr_sets(data, condition)
    l_branch = build_tree(l_set, depth + 1, max_depth, min_set_size, method)
    r_branch = build_tree(r_set, depth + 1, max_depth, min_set_size, method)

    return Node(condition, l_branch, r_branch)


def print_tree(node, spacing=""):
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.rate)
        return

    print(spacing + str(node.condition))

    print(spacing + 'IF True:')
    print_tree(node.l_branch, spacing + "*------*")

    print(spacing + 'IF False:')
    print_tree(node.r_branch, spacing + "*------*")
