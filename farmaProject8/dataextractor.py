import pandas as pd
import multiprocessing as mlpt
import math
from model import User


def _load_train_datasets():
    df = pd.read_csv('data/train.csv', sep=';')
    df.drop(columns=['idx'], inplace=True)
    val_df = df.sample(n=7800, random_state=21)
    df.drop(index=val_df.index, inplace=True)
    return df.values.tolist(), val_df.values.tolist()


def _load_task_datasets():
    train_df = pd.read_csv('data/train.csv', sep=';')
    train_df.drop(columns=['idx'], inplace=True)
    test_df = pd.read_csv('data/task.csv', sep=',')
    return train_df.values.tolist(), test_df.values.tolist()


def _create_user_list(df_val: list):
    unique_users = list(set([x[0] for x in df_val]))
    return [User(usr) for usr in unique_users]


def _add_movies_for_usr(train_set: list, users: list, lst):
    for user in users:
        for row in train_set:
            if row[0] == user.user_id:
                user.movies.append((row[1], row[2]))

        lst.append(user)


def _create_users_multi(n_threads: int, train_set: list):
    manager = mlpt.Manager()
    processes = []
    final_users = manager.list()
    mlpt_users = [manager.list() for _ in range(n_threads)]
    users = _create_user_list(train_set)
    step = math.floor(len(users) / (n_threads - 1))
    usr_rng = range(0, len(users) - step, step)
    for i, val in enumerate(usr_rng):
        mlpt_users[i] = users[val:(val + step)]
        processes.append(mlpt.Process(target=_add_movies_for_usr, args=(train_set, mlpt_users[i], final_users)))
        if i == len(usr_rng) - 1:
            mlpt_users[n_threads - 1] = users[val + step:(val + step + (len(users) % n_threads))]
            processes.append(mlpt.Process(target=_add_movies_for_usr, args=(train_set, mlpt_users[n_threads - 1],
                                                                            final_users)))

    for i in range(n_threads):
        processes[i].start()
    for i in range(n_threads):
        processes[i].join()

    return list(final_users)


def get_data_train():
    train_set, val_set = _load_train_datasets()
    mlpt_users = _create_users_multi(12, train_set)
    return mlpt_users, val_set


def get_data_task():
    train_set, test_set = _load_task_datasets()
    mlpt_users = _create_users_multi(12, train_set)
    return mlpt_users, test_set
