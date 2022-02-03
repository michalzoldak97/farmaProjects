import random
import pandas as pd


def _load_raw_csv():
    train_df = pd.read_csv('../data/train.csv', sep=';')
    task_df = pd.read_csv('../data/task.csv', sep=',')
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


def load_train_val_data(val_n=7800):
    train_df, _ = _load_raw_csv()
    train_df.drop(columns=['idx'], inplace=True)
    val_df = train_df.sample(n=val_n, random_state=42)
    train_df.drop(index=val_df.index, inplace=True)
    return train_df, val_df


def load_task_data():
    train_df, task_df = _load_raw_csv()
    train_df.drop(columns=['idx'], inplace=True)
    task_df.drop(columns=['idx'], inplace=True)
    return train_df, task_df
