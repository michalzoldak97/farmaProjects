from infogain import calc_gini_idx, coverage_measure
import pandas as pd
from datamodifier import test_train_split, test_train_pca, load_test_train_pca, load_main_test_train
from tree import build_tree, FinalLeaf, print_tree
import datetime
import time
import os
import multiprocessing as mlpt


def get_most_common_idx(pred: list):
    h_val = 0
    cls = 0
    for idx, val in enumerate(pred):
        if val > h_val:
            h_val = val
            cls = idx

    return float(cls)


def tree_classify(row, node):
    if isinstance(node, FinalLeaf):
        return get_most_common_idx(node.predictions)

    if node.condition.match(row):
        return tree_classify(row, node.true_group)
    else:
        return tree_classify(row, node.false_group)


def tree_class_print(row, node):
    if isinstance(node, FinalLeaf):
        return get_most_common_idx(node.predictions)

    if node.condition.match(row):
        return tree_classify(row, node.true_group)
    else:
        return tree_classify(row, node.false_group)


def classify_single():
    stat_f = open('results/resultt5.txt', 'w')
    star_time = datetime.datetime.now()
    init_start_time = datetime.datetime.now()
    test_y, train_set, test_set = load_test_train_pca(version='all', num=0)
    stat_f.write("Data loading took: {} \n".format(datetime.datetime.now() - star_time))
    star_time = datetime.datetime.now()
    cls_tree = build_tree(train_set, 0, max_depth=19, min_size=50)
    stat_f.write("Tree build took: {} \n".format(datetime.datetime.now() - star_time))
    star_time = datetime.datetime.now()
    y_pred = []
    for row in test_set:
        y_pred.append(tree_classify(row, cls_tree))
    stat_f.write("Classification took: {} \n".format(datetime.datetime.now() - star_time))
    stat_f.write("Execution took: {} \n".format(datetime.datetime.now() - init_start_time))
    print("Execution time: {}".format(datetime.datetime.now() - init_start_time))
    print(coverage_measure(y_pred, test_y))
    stat_f.write("Result:\n{} ".format(coverage_measure(y_pred, test_y)))

    # time.sleep(2)
    # os.system('systemctl poweroff')


def console_print_tree():
    test_y, train_set, test_set = load_test_train_pca(version='all', num=0)
    cls_tree = build_tree(train_set, 0, max_depth=23, min_size=50)
    print_tree(cls_tree, "||")


def multi_task(idx: int, m_list):
    stat_f_path = 'results/result_8_' + str(idx) + '.txt'
    stat_f = open(stat_f_path, 'w')
    star_time = datetime.datetime.now()
    test_y, train_set, test_set = load_test_train_pca(version='all', num=1)
    cls_tree = build_tree(train_set, 0, max_depth=13 + (idx + 2), min_size=50)
    for row in test_set:
        m_list.append(tree_classify(row, cls_tree))
    stat_f.write("Process num: {} \n".format(idx))
    stat_f.write("max_depth: {}  min_size: {} set: {}\n".format(13 + (idx + 2), 50, 13))
    stat_f.write("Execution took: {} \n".format(datetime.datetime.now() - star_time))
    stat_f.write("Result:\n{} ".format(coverage_measure(list(m_list), test_y)))


def classify_multi(n_threads: int):
    manager = mlpt.Manager()
    mlpt_lists = [manager.list() for _ in range(n_threads)]
    processes = [mlpt.Process(target=multi_task, args=(i, mlpt_lists[i])) for i in range(n_threads)]

    for i in range(n_threads):
        processes[i].start()
    for i in range(n_threads):
        processes[i].join()


def final_classify():
    train_set, test_set = load_main_test_train()
    cls_tree = build_tree(train_set, 0, max_depth=19, min_size=50)
    y_pred = [int(tree_classify(row, cls_tree)) for row in test_set]
    submission_df = pd.read_csv('data/task.csv', sep=';')
    submission_df['rate'] = y_pred
    submission_df.to_csv('submissions/submission_2.csv', sep=';', index=False)
    # print_tree(cls_tree, "||")


# if __name__ == "__main__":
#     classify_multi(n_threads=12)
# classify_single()
# console_print_tree()
final_classify()
