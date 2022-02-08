import pandas as pd


def _load_raw_csv():
    train_df = pd.read_csv('../data/train.csv', sep=';')
    task_df = pd.read_csv('../data/task.csv', sep=',')
    return train_df, task_df


def _load_train_val_data(val_n=7800):
    train_df, _ = _load_raw_csv()
    train_df.drop(columns=['idx'], inplace=True)
    val_df = train_df.sample(n=val_n, random_state=42)
    train_df.drop(index=val_df.index, inplace=True)

    return train_df, val_df


def _normalize_m_feats(m_feats: pd.DataFrame):
    for col in m_feats.columns[1:]:
        m_feats[col] = m_feats[col] / m_feats[col].abs().max()
    m_feats = m_feats.values.tolist()
    for i, row in enumerate(m_feats):
        m_feats[i][0] = int(row[0])

    return m_feats


def _replace_m_id_with_feats(df_to_modify: pd.DataFrame, m_feats: list):
    to_modify = df_to_modify.values.tolist()
    for mov in m_feats:
        for set_row in to_modify:
            if set_row[1] == mov[0]:
                set_row[1] = mov[1:]

    print(to_modify)

    return to_modify

# movie_feats_df.drop(columns=['has_collection', 'popularity', 'budget', 'language', 'runtime', 'revenue',
#                         'vote_avg', 'vote_count', 'gen_one', 'gen_two', 'gen_three', 'gen_four'], inplace=True)

def get_train_val_data():
    train_df, val_df = _load_train_val_data()
    movie_feats_df = pd.read_csv('../data/movie_collection_data_4.csv')
    movie_feats_df.drop(columns=['has_collection', 'language', 'runtime',
                        'vote_avg', 'gen_one', 'gen_two', 'gen_three', 'gen_four'], inplace=True)
    movie_feats_df = _normalize_m_feats(movie_feats_df)
    train_set = _replace_m_id_with_feats(train_df, movie_feats_df)
    val_set = _replace_m_id_with_feats(val_df, movie_feats_df)
    return train_set, val_set
