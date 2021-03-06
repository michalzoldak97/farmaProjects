from userModel import User
from dataextractor import get_train_val_data, get_train_task_data
from distances import coverage_measure
import pandas as pd


def _build_user_list(train_set: list):
    all_users = {}
    for train_row in train_set:
        if train_row[0] in all_users.keys():
            movie_dict = {'features': train_row[1], 'rate': train_row[2]}
            all_users[train_row[0]].movies.append(movie_dict)
        else:
            all_users[train_row[0]] = User(train_row[0])

    return all_users


def _build_results_list(test_set: list, all_users: dict, k):
    y_true = [row[-1] for row in test_set]
    y_res = []
    for t_row in test_set:
        for user_id, user in all_users.items():
            if t_row[0] == user_id:
                y_res.append(user.get_rate_for_new_movie(t_row[1], k))

    return y_res, y_true


def classify_val(k: int):
    train_set, val_set = get_train_val_data()
    all_users = _build_user_list(train_set)
    y_res, y_true = _build_results_list(val_set, all_users, k)
    print(coverage_measure(y_res, y_true))


def classify_task(k: int):
    train_set, task_set = get_train_task_data()
    all_users = _build_user_list(train_set)
    y_res, _ = _build_results_list(task_set, all_users, k)
    sub = pd.read_csv('../data/task.csv', sep=';')
    sub.drop(columns=['rate'], inplace=True)
    sub['rate'] = y_res
    sub.to_csv('../submissions/submission_1.csv', index=False, sep=';')


k_ = 30
# classify_val(k_)
classify_task(k_)
