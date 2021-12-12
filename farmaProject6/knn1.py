from distances import euc_dist, cos_dist, acc_measure
import pandas as pd
from sklearn.metrics import accuracy_score
import multiprocessing as mlpt
import datetime
import sys


def split_dataset(dataset):
    tr_df = pd.read_csv(dataset)
    tr_df.drop(columns=['has_collection', 'popularity', 'budget', 'language', 'runtime', 'revenue', 'vote_avg',
                        'vote_count', 'gen_one', 'gen_two', 'gen_three', 'gen_four'], inplace=True)
    for col in tr_df:
        if col != 'rate':
            tr_df[col] = ((tr_df[col] - tr_df[col].min()) / (tr_df[col].max() - tr_df[col].min()))

    # tr_df = (tr_df - tr_df.mean()) / tr_df.std()
    # print(tr_df.head(20))
    # sys.exit()
    tr_df.user = tr_df.user * 10
    test_set = tr_df.sample(n=7800)
    tr_df.drop(index=test_set.index, inplace=True)
    tr_set = tr_df.values.tolist()
    tr_y = [row[-1] for row in tr_set]
    tr_x = [row[:-1] for row in tr_set]
    y_test = [row[-1] for row in test_set.values.tolist()]
    test_set.drop(columns=['rate'], inplace=True)
    x_tst = test_set.values.tolist()
    return tr_set, tr_x, tr_y, x_tst, y_test


def predict_dataset(datase)


def get_neigb(tr_set, test_row, num):
    dist = [(tr_row, euc_dist(test_row, tr_row[:-1])) for tr_row in tr_set]
    dist.sort(key=lambda tup: tup[1])
    return [dist[i][0] for i in range(num)]


def classify(tr_set, test_row, num):
    neigbs = get_neigb(tr_set, test_row, num)
    out = [row[-1] for row in neigbs]
    return max(set(out), key=out.count)


train_set, x_train, y_train, x_test, y_target = split_dataset('train_data_1.csv')
y_pred = []
star_time = datetime.datetime.now()

manager = mlpt.Manager()


def multi_task(x_vec, lst):
    lst.append(classify(train_set, x_vec, 10))


def main_task():
    global y_pred
    train_data_parts = []
    threads = 12
    part = 650
    for _ in range(threads):
        train_data_parts.append(manager.list())

    for idx in range(part):
        processes = []
        for i in range(threads):
            processes.append(mlpt.Process(target=multi_task, args=(x_test[idx + (i * part)], train_data_parts[i])))
        for i in range(threads):
            processes[i].start()
        for i in range(threads):
            processes[i].join()

    for i in range(threads):
        y_pred = y_pred + list(train_data_parts[i])


if __name__ == "__main__":
    main_task()

print("Execution time: {}".format(datetime.datetime.now() - star_time))
print(accuracy_score(y_target, y_pred))
print(acc_measure(y_pred, y_target)) # 0.529
