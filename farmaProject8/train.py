from dataextractor import get_data_train, get_data_task
from model import User
from math import sqrt, floor
import datetime
import multiprocessing as mlpt
import pandas as pd


def euc_dist(r1, r2):
    dist = .0
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

    return cov_str


def _get_usr_and_similar(row: list):
    curr_user = None
    all_users = []
    for user in users:
        if user.has_a_movie(row[1]):
            all_users.append(user)
    for user in users:
        if user.user_id == row[0]:
            curr_user = user
            break

    return curr_user, all_users


def _rank_users(usr: User, others: list, max_u: int):
    ranking = []
    for user in others:
        rank_row = [user, .0]
        for movie in usr.movies:
            for mv in user.movies:
                if movie[0] == mv[0]:
                    diff = euc_dist([movie[1]], [mv[1]])
                    if diff == 0:
                        rank_row[1] += 1.
                    else:
                        rank_row[1] += 1. / float(diff)
        ranking.append(rank_row)

    ranking.sort(key=lambda x: x[1], reverse=True)
    return [ranking[i][0] for i in range(max_u)]


def _get_rank(mov_idx: int, usr_lst: list):
    votes = []
    for user in usr_lst:
        votes.append(user.get_movie(mov_idx)[1])

    return max(set(votes), key=votes.count)


def _classify_row(row: list, max_u=5):
    curr_user, all_users = _get_usr_and_similar(row)
    rank = _rank_users(curr_user, all_users, max_u)
    return _get_rank(row[1], rank)


def _mlpt_proc(set_: list, lst):
    for row_x in set_:
        lst.append(_classify_row(row_x, 5))


def _get_range(val_x: list, n_threads: int):
    step = floor(len(val_x) / n_threads)
    val_rng = [x for x in range(0, len(val_x) - step, step)]
    last_max = len(val_x) % n_threads
    val_rng.append(val_rng[-1] + step + last_max)

    for i, val in enumerate(val_rng):
        up = step
        if i == len(val_rng) - 1:
            up = last_max
            if up == 0:
                up = step
        val_rng[i] = (val, val + up)

    return val_rng


def predict_multi(n_threads: int):
    global val_set
    val_x = [row[1:3] for row in val_set]
    p_range = _get_range(val_x, n_threads)

    manager = mlpt.Manager()
    m_lists = [manager.list() for _ in range(n_threads)]
    processes = []
    for i, val in enumerate(p_range):
        processes.append(mlpt.Process(target=_mlpt_proc, args=(val_x[val[0]:val[1]], m_lists[i])))

    for i in range(n_threads):
        processes[i].start()
    for i in range(n_threads):
        processes[i].join()

    y_pred = [list(lst) for lst in m_lists]
    y_pred = [val for lst in y_pred for val in lst]
    return y_pred


def predict_train():
    global val_set
    val_x = [row[:2] for row in val_set]
    val_true = [row[2] for row in val_set]
    y_pred = [_classify_row(row, 5) for row in val_x]
    print(coverage_measure(y_pred, val_true))


def predict_test():
    y_pred = predict_multi(12)
    y_pred = [int(x) for x in y_pred]
    submission_df = pd.read_csv('data/task.csv', sep=';')
    submission_df['rate'] = y_pred
    submission_df.to_csv('data/submission_1.csv', sep=';', index=False)


users, val_set = get_data_task()
star_time = datetime.datetime.now()
if __name__ == '__main__':
    predict_test()
print("Execution time: {}".format(datetime.datetime.now() - star_time))

