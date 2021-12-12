from distances import euc_dist
import pandas as pd
import multiprocessing as mlpt
import datetime


def get_train_test_set():
    tr_df = pd.read_csv('train_data_1.csv')
    tr_df.drop(columns=['has_collection', 'popularity', 'budget', 'language', 'runtime', 'revenue', 'vote_avg',
                        'vote_count'], inplace=True)
    for col in tr_df:
        if col != 'rate':
            tr_df[col] = ((tr_df[col] - tr_df[col].min()) / (tr_df[col].max() - tr_df[col].min()))

    tr_df.user = tr_df.user * 10000
    tr_df.movie = tr_df.movie * 2

    tst_df = pd.read_csv('task.csv', delimiter=';')
    mov_feat_df = pd.read_csv('movie_collection_data_4.csv')
    tst_df.insert(2, 'gen_one', 0.0)
    tst_df.insert(3, 'gen_two', 0.0)
    tst_df.insert(4, 'gen_three', 0.0)
    tst_df.insert(5, 'gen_four', 0.0)
    for idx in tst_df.index:
        movie = tst_df.at[idx, 'movie']
        for col in tst_df:
            try:
                tst_df.at[idx, col] = float(mov_feat_df.at[movie, col])
            except KeyError:
                continue
    tst_df.drop(columns=['idx', 'rate'], inplace=True)
    tst_df.user = pd.to_numeric(tst_df.user, downcast='float')
    for idx in tst_df.index:
        tst_df.at[idx, 'user'] = ((float(tst_df.at[idx, 'user']) * 1.0) / 1816.0) * 100.0
    for col in tst_df:
        if col != 'rate':
            tst_df[col] = ((tst_df[col] - tst_df[col].min()) / (tst_df[col].max() - tst_df[col].min()))
    tst_df.user = tst_df.user * 10000
    tst_df.movie = tst_df.movie * 2

    return tr_df.values.tolist(), tst_df.values.tolist()


def get_neigb(tr_set, test_row, num):
    dist = [(tr_row, euc_dist(test_row, tr_row[:-1])) for tr_row in tr_set]
    dist.sort(key=lambda tup: tup[1])
    return [dist[i][0] for i in range(num)]


def classify(tr_set, test_row, num):
    neigbs = get_neigb(tr_set, test_row, num)
    out = [row[-1] for row in neigbs]
    return max(set(out), key=out.count)


train_set, test_set = get_train_test_set()
y_pred = []
star_time = datetime.datetime.now()

manager = mlpt.Manager()


def multi_task_loop(lst, idx_range):
    for idx in range(idx_range[0], idx_range[1]):
        lst.append(classify(train_set, test_set[idx], 20))


def main_task():
    global y_pred
    train_data_parts = []
    threads = 12
    part = 895
    for _ in range(threads):
        train_data_parts.append(manager.list())

    processes = []
    for i in range(threads):
        processes.append(mlpt.Process(target=multi_task_loop, args=(train_data_parts[i], (i * part, (i + 1) * part))))
    for i in range(threads):
        processes[i].start()
    for i in range(threads):
        processes[i].join()

    for i in range(threads):
        y_pred = y_pred + list(train_data_parts[i])


if __name__ == "__main__":
    main_task()

test_df = pd.read_csv('task.csv', delimiter=';')
test_df.drop(columns=['rate'], inplace=True)
y_pred = [int(y) for y in y_pred]
test_df['rate'] = y_pred

test_df.to_csv('submission4.csv', sep=';', index=False)


print("Execution time: {}".format(datetime.datetime.now() - star_time))