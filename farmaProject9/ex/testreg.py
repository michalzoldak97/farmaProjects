import scipy.io
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from numpy import random
import pickle
import timeit
import dataloader as d


# sample_data = d.load_sample(5, 5)

def _calc_func_val(args, x_all):
    res = .0
    for i, x in enumerate(args[:-1]):
        res += x * x_all[i]

    return res + args[-1]


def play_ground():
    n_feat = 3
    epochs = range(1000)
    _lr = .01
    sample_data = pd.DataFrame()
    sample_data['user'] = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    sample_data['movie'] = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    sample_data['rating'] = [5, 4, 3, 2, 1, 5, 4, 3, 2, 1, 5, 4, 3, 2, 1]

    unique_users_num = len(sample_data['user'].unique())
    unique_movies_num = len(sample_data['movie'].unique())

    user_movie_mtx = np.zeros((unique_movies_num, unique_users_num))

    for u in range(unique_users_num):
        for m in range(unique_movies_num):
            df = sample_data[(sample_data['user'] == u) & (sample_data['movie'] == m)]
            if df.empty:
                continue
            user_movie_mtx[m][u] = df['rating']

    m_feats = [[1. for _ in range(n_feat)] for _ in range(unique_movies_num)]
    u_params = [[1. for _ in range(n_feat + 1)] for _ in range(unique_users_num)]

    # movie features
    for epoch in epochs:
        for u, vec in enumerate(u_params):
            loss_u = [0. for _ in vec]
            # predictions for existing user movies
            y_pred = [_calc_func_val(vec, col) for col in m_feats]
            # actual values for existing user movies
            ratings = [user_movie_mtx[i][u] for i, _ in enumerate(user_movie_mtx)]
            y_diff = [ratings[i] - pred for i, pred in enumerate(y_pred)]
            for i, arg in enumerate(vec[:-1]):
                grad_m = (1. / 2.) * sum([diff * m_feats[j][i] for j, diff in enumerate(y_diff)])
                loss_u[i] = grad_m
                u_params[u][i] = arg + (_lr * grad_m)

            grad_c = (1. / 2.) * sum(y_pred)
            loss_u[-1] = grad_c
            u_params[u][len(vec) - 1] = u_params[u][len(vec) - 1] + (_lr * grad_c)

            print("User {} Loss {} VM {}".format(u, loss_u, sum(loss_u)))

        for m, vec in enumerate(m_feats):
            loss_m = [0. for _ in vec]
            y_pred = [_calc_func_val(params, vec) for params in u_params]
            ratings = [user_movie_mtx[m][i] for i, _ in enumerate(u_params)]
            y_diff = [ratings[i] - pred for i, pred in enumerate(y_pred)]
            for i, feat in enumerate(vec):
                grad_m = (1. / 2.) * sum([diff * u_params[j][i] for j, diff in enumerate(y_diff)])
                loss_m[i] = grad_m
                m_feats[m][i] = feat + (_lr * grad_m)

            print("Movie {} Loss {} VM {}".format(m, loss_m, sum(loss_m)))





    print(unique_users_num, unique_movies_num)
    print(sample_data)
    print(user_movie_mtx)
    print(m_feats)
    print(u_params)


play_ground()


# R = np.zeros((unique_users_num, unique_movies_num))
# for line in sample_data.itertuples():
#     print(line[1], line[2], line[3])
#     R[line[1] - 1, line[2] - 1] = line[3]
#
# n_u = n_m = 100
# R = R[:n_u, :n_m]
#
# print(R)
#
# users, movies = R.nonzero()
# users = np.unique(users)
# movies = np.unique(movies)
#
# print(users)
# print(movies)
