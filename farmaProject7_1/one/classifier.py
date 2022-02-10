from user import User
from dataextractor import get_train_val_data, get_train_task_data
from distances import coverage_measure
import pandas as pd


# hiperparams
# max_depth, min_size, method = 3, 50, 'gini'


def _build_user_list(train_set: list, max_depth=3, min_size=50, method='gini'):
    all_users = {}
    for train_row in train_set:
        if train_row[0] in all_users.keys():
            movie_dict = {'features': train_row[1], 'rate': train_row[2]}
            all_users[train_row[0]].movies.append(movie_dict)
        else:
            all_users[train_row[0]] = User(train_row[0], max_depth, min_size, method)

    for user_idx in all_users:
        all_users[user_idx].build_user_tree()

    all_users[1641].print_tree()
    return all_users


def _build_results_list(test_set: list, all_users: dict):
    y_true = [row[-1] for row in test_set]
    y_res = []
    for t_row in test_set:
        for user_id, user in all_users.items():
            if t_row[0] == user_id:
                y_res.append(user.get_rate_for_new_movie(t_row[1]))

    return y_res, y_true


def classify_val(feats_to_use: list, h_params=None):
    if h_params is not None:
        max_depth = h_params['max_depth']
        min_size = h_params['min_size']
        method = h_params['method']
    train_set, val_set = get_train_val_data(feats_to_use)
    all_users = _build_user_list(train_set, max_depth, min_size, method)
    y_res, y_true = _build_results_list(val_set, all_users)
    cov_measure = coverage_measure(y_res, y_true)
    # print("Cov measure: {} \n for max: {}, min: {}, method: {}".format(cov_measure[0], max_depth, min_size, method))
    return cov_measure[1], cov_measure[2]


def classify_task(feats_to_use: list):
    train_set, val_set = get_train_task_data(feats_to_use)
    all_users = _build_user_list(train_set, 5, 60, 'entropy')
    y_res, _ = _build_results_list(val_set, all_users)
    sub = pd.read_csv('../data/task.csv', sep=';')
    sub.drop(columns=['rate'], inplace=True)
    sub['rate'] = y_res
    sub.to_csv('../submissions/submission_1.csv', index=False, sep=';')


#
feats = ['language', 'popularity', 'vote_avg', 'vote_count', 'gen_two', 'gen_three', 'gen_four']
classify_task(feats)
# classify_val(feats, {'max_depth': 3, 'min_size': 30, 'method': 'gini'})
