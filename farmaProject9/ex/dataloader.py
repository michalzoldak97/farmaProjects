import random
import pandas as pd


def _load_raw_csv():
    train_df = pd.read_csv('../data/train.csv', sep=';')
    task_df = pd.read_csv('../data/task.csv', sep=';')
    return train_df, task_df


def _get_random_from_unique(df, col, num):
    u_ids = pd.unique(df[col])
    return random.choices(u_ids.tolist(), k=num)


def load_sample(num_u=2, num_m=5):
    train_df, _ = _load_raw_csv()
    user_ids = _get_random_from_unique(train_df, 'user', num_u)
    train_df = train_df[(train_df['user'].isin(user_ids))]
    movie_ids = _get_random_from_unique(train_df, 'movie', num_m)
    train_df = train_df[(train_df['movie'].isin(movie_ids))]
    return train_df.drop(columns=['idx'])

