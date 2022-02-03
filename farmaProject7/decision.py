from infogain import calc_gini_idx, coverage_measure
from datamodifier import test_train_split, test_train_pca, load_short_test_train_pca
import datetime
import multiprocessing as mlpt


def eval_split(idx, val, ds):
    l, r = list(), list()
    for row in ds:
        if row[idx] < val:
            l.append(row)
        else:
            r.append(row)

    return l, r


def get_best_split(ds):
    class_vals = list(set(row[-1] for row in ds))
    b_idx, b_val, b_score, b_gr = 999, 999, 999, None
    for idx in range(len(ds[0]) - 1):
        for row in ds:
            gr = eval_split(idx, row[idx], ds)
            info_idx = calc_gini_idx(gr, class_vals)
            if info_idx < b_score:
                b_idx, b_val, b_score, b_gr = idx, row[idx], info_idx, gr

    return {'idx': b_idx, 'val': b_val, 'groups': b_gr}


def to_leaf(gr):
    out = [row[-1] for row in gr]
    return max(set(out), key=out.count)


def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    if not left or not right:
        node['left'] = node['right'] = to_leaf(left + right)
        return
    if depth >= max_depth:
        node['left'] = node['right'] = to_leaf(left), to_leaf(right)
        return

    if len(left) <= min_size:
        node['left'] = to_leaf(left)
    else:
        node['left'] = get_best_split(left)
        split(node['left'], max_depth, min_size, depth + 1)

    if len(right) <= min_size:
        node['right'] = to_leaf(right)
    else:
        node['right'] = get_best_split(right)
        split(node['right'], max_depth, min_size, depth + 1)


def build_tree(train, max_depth, min_size, ):
    root = get_best_split(train)
    split(root, max_depth, min_size, 1)
    return root


def predict(node, row):
    if row[node['idx']] < node['val']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


def decision_tree(train, test, max_depth, min_size):
    print(len(train))
    print(len(test))
    tree = build_tree(train, max_depth, min_size)
    predictions = list()
    for i, row in enumerate(test):
        prediction = predict(tree, row)
        predictions.append(prediction)
        print("Row: {}".format(i))

    return predictions


def classify_single():
    star_time = datetime.datetime.now()
    test_y, all_train, all_test = load_short_test_train_pca()
    y_pred = decision_tree(all_train, all_test, 20, 2)
    for i, y in enumerate(y_pred):
        if isinstance(y, tuple):
            y_pred[i] = y[0]
    print("Execution time: {}".format(datetime.datetime.now() - star_time))
    print(coverage_measure(y_pred, test_y))


y_pred = []


def m_task(tr_set: list, test_set: list, lst: list):
    lst.append(decision_tree(tr_set, test_set, 20, 10))


def classify_multi():
    global y_pred
    train_x, train_y, test_x, test_y, all_train, all_test = test_train_split()
    test_data_parts, processes = [], []
    threads = 12
    print(len(all_test) / 12)


classify_single()
