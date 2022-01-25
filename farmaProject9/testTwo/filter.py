import math
import pandas as pd
import dataloader as d
import datetime
import multiprocessing as mlpt
import matplotlib.pyplot as plt


def _get_range(len_set: int, n_threads: int):
    step = math.floor(len_set / n_threads)
    val_rng = [x for x in range(0, len_set - step, step)]
    last_max = len_set % n_threads
    val_rng.append(val_rng[-1] + step)

    for i, val in enumerate(val_rng):
        up = step
        if i == len(val_rng) - 1:
            up = last_max
            if up == 0:
                up = step
        val_rng[i] = (val, val + up)

    return val_rng


class DataExtractor:
    def __init__(self, n_feat: int, mode='train'):
        self.users_set = dict()
        self.n_feat = n_feat
        self.mode = mode

    def _set_u_dict(self, df: pd.DataFrame):
        u_users = df['user'].unique().tolist()
        for i, u in enumerate(u_users):
            self.users_set[i] = u

        return len(u_users)

    def _switch_df_values(self, df: pd.DataFrame):
        for kv in self.users_set.items():
            df.loc[(df['user'] == kv[1]), 'user'] = kv[0]

        df['movie'] = df['movie'].apply(lambda x: x - 1)
        return df

    def _transform_indexes(self, t_df: pd.DataFrame, v_df: pd.DataFrame):
        self.uu_n = self._set_u_dict(t_df)
        self.um_n = len(t_df['movie'].unique())

        t_df = self._switch_df_values(t_df)
        v_df = self._switch_df_values(v_df)
        return t_df, v_df

    def _scan_set_by_user(self, usr_rng: tuple, t_df: pd.DataFrame, um_list: list):
        for usr in range(usr_rng[0], usr_rng[1]):
            user_movie_lst = []
            for m in range(self.um_n):
                df = t_df[(t_df['user'] == usr) & (t_df['movie'] == m)]
                if df.empty:
                    user_movie_lst.append(.0)
                else:
                    user_movie_lst.append(df['rate'].values.tolist()[0])
            um_list.append(user_movie_lst)

    def _fill_um_matrix_multi(self, t_df: pd.DataFrame, n_threads: int):
        manager = mlpt.Manager()
        m_lists = [manager.list() for _ in range(n_threads)]
        processes = []
        p_range = _get_range(self.uu_n, n_threads - 1)
        for i, val in enumerate(p_range):
            processes.append(mlpt.Process(target=self._scan_set_by_user, args=(val, t_df, m_lists[i])))

        for i in range(n_threads):
            processes[i].start()
        for i in range(n_threads):
            processes[i].join()

        matrix = [[] for _ in range(self.um_n)]
        for lst in m_lists:
            for um_list in list(lst):
                for i, um in enumerate(um_list):
                    matrix[i].append(um)

        return matrix

    def _transform_data(self):
        t_df, v_df = d.load_train_val_data()
        train_df, self.val_df = self._transform_indexes(t_df, v_df)

        self.user_movie_mtx = self._fill_um_matrix_multi(train_df, 12)
        self.m_feats = [[1. for _ in range(self.n_feat)] for _ in range(self.um_n)]
        self.u_params = [[1. for _ in range(self.n_feat + 1)] for _ in range(self.uu_n)]

    def get_train_val_data(self):
        self._transform_data()
        return {
            "users_set": self.users_set,
            "user_movie_matrix": self.user_movie_mtx,
            "m_feats": self.m_feats,
            "u_params": self.u_params,
            "val_df": self.val_df
        }


class CollFilter:
    def __init__(self, n_feat: int, epochs: int, lr_: float, lmb: float, mode='train'):
        self.n_feat = n_feat
        self.epochs = range(epochs)
        self.lr_ = lr_
        self.lmb = lmb
        self.u_loss = [[] for _ in self.epochs]
        if mode == 'train':
            dx = DataExtractor(n_feat)
            self.users_set, self.user_movie_mtx, self.m_feats, self.u_params, self.val_df = dx.get_train_val_data()\
                .values()
            print(self.val_df.head())

    def _calc_func_val(self, args, x_all):
        res = .0
        for i, x in enumerate(args[:-1]):
            res += x * x_all[i]

        return res + args[-1]

    def _minimize(self):
        for epoch in self.epochs:
            for u, vec in enumerate(self.u_params):
                loss_u = [0. for _ in vec]
                n = 0.
                ratings = [self.user_movie_mtx[i][u] for i, _ in enumerate(self.user_movie_mtx)]
                y_pred, y_diff = [], []
                for m_idx, r in enumerate(ratings):
                    if r != .0:
                        f_val = self._calc_func_val(vec, self.m_feats[m_idx])
                        y_pred.append(f_val)
                        y_diff.append(r - f_val)
                        n += 1.
                    else:
                        y_pred.append(None)
                        y_diff.append(None)

                if n < 1:
                    continue

                for i, arg in enumerate(vec[:-1]):
                    grad_m = 0
                    for j, diff in enumerate(y_diff):
                        if diff is not None:
                            grad_m += diff * self.m_feats[j][i]
                            # print("Grad {} += {} + {}".format(grad_m, diff, self.m_feats[j][i]))

                    grad_m = grad_m / n
                    grad_m = grad_m + (self.lmb * arg)
                    loss_u[i] = grad_m
                    # print("Param is {}".format(self.u_params[u][i]))
                    self.u_params[u][i] = arg + (self.lr_ * grad_m)
                    # print("After param is {}".format(self.u_params[u][i]))

                grad_c = 0
                for diff in y_diff:
                    if diff is not None:
                        grad_c += diff
                grad_c = grad_c / n
                grad_c = grad_c + (self.lmb * self.u_params[u][self.n_feat])
                loss_u[-1] = grad_c
                self.u_params[u][self.n_feat] = self.u_params[u][self.n_feat] * (self.lr_ + grad_c)

                print("Epoch: {} User {} VM {}".format(epoch, u, sum(loss_u)))
                self.u_loss[epoch].append(sum(loss_u))

            for m, vec in enumerate(self.m_feats):
                c = 0.
                loss_m = [0. for _ in vec]
                ratings = [self.user_movie_mtx[m][i] for i, _ in enumerate(self.u_params)]
                y_pred, y_diff = [], []
                for u_idx, r in enumerate(ratings):
                    if r != .0:
                        f_val = self._calc_func_val(self.u_params[u_idx], vec)
                        y_pred.append(f_val)
                        y_diff.append(r - f_val)
                        c += 1.
                    else:
                        y_pred.append(None)
                        y_diff.append(None)

                if c < 1:
                    continue

                for i, feat in enumerate(vec):
                    grad_n = 0
                    for j, diff in enumerate(y_diff):
                        if diff is not None:
                            grad_n += diff * self.u_params[j][i]
                    grad_n = grad_n / c
                    grad_n = grad_n + (self.lmb * feat)
                    loss_m[i] = grad_n
                    self.m_feats[m][i] = feat + (self.lr_ * grad_n)

                print("Epoch: {} Movi {} VM {}".format(epoch, m, sum(loss_m)))

    def train(self):
        star_time = datetime.datetime.now()
        self._minimize()
        print("Execution time: {}".format(datetime.datetime.now() - star_time))
        self.u_loss = [sum(losses) for losses in self.u_loss]
        x = [i for i in self.epochs]
        plt.scatter(x, self.u_loss)
        plt.show()


train_filter = CollFilter(3, 1000, .01, .01)
train_filter.train()
