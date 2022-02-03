import pandas as pd
import dataloader as d


class CollFilter:
    def __init__(self, n_feat: int, epochs: int, lr_: float, lmb: float):
        self.users_set = dict()
        self.n_feat = n_feat
        self.epochs = range(epochs)
        self.lr_ = lr_
        self.lmb = lmb

    def _calc_func_val(self, args, x_all):
        res = .0
        for i, x in enumerate(args[:-1]):
            res += x * x_all[i]

        return res + args[-1]

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

    def _transform_data(self):
        t_df, v_df = d.load_train_val_data()
        train_df, val_df = self._transform_indexes(t_df, v_df)
        user_movie_mtx = [[.0 for _ in range(self.uu_n)] for _ in range(self.um_n)]

        for u in range(self.uu_n):
            for m in range(self.um_n):
                df = train_df[(train_df['user'] == u) & (train_df['movie'] == m)]
                if df.empty:
                    continue
                user_movie_mtx[m][u] = df['rate'].values.tolist()[0]

        self.user_movie_mtx = user_movie_mtx
        self.m_feats = [[1. for _ in range(self.n_feat)] for _ in range(self.um_n)]
        self.u_params = [[1. for _ in range(self.n_feat + 1)] for _ in range(self.uu_n)]

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
                # break

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
                # break
            # break

    def train(self):
        self._transform_data()
        self._minimize()


train_filter = CollFilter(3, 1000, .01, .01)
train_filter.train()
