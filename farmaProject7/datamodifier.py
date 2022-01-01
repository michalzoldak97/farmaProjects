import pandas as pd
from sklearn.decomposition import PCA
import sys


def _generate_set(pth: str, sep=';'):
    df = pd.read_csv(pth, sep=sep)
    feat_df = pd.read_csv('data/movie_collection_data_4.csv')
    df.drop(columns=['idx'], inplace=True)

    feat_cols = [col_name for col_name in feat_df]

    for idx in range(2, 14):
        df.insert(idx, feat_cols[idx - 1], .0)

    feat_cols.pop(0)

    for idx in df.index:
        m_feat = df.at[idx, 'movie'] - 1
        for f_col in feat_cols:
            df.at[idx, f_col] = feat_df.at[m_feat, f_col]

    for col in df:
        pd.to_numeric(df[col], downcast='float')

    print(len(df.index))
    return df


def _create_train_test():
    train_df = _generate_set('data/train.csv')
    task_df = _generate_set('data/task.csv', ',')
    train_df.to_csv('data/main_train_1', index=False)
    task_df.to_csv('data/main_test_1', index=False)


def load_main_test_train():
    train_set = pd.read_csv('data/main_train_1')
    test_set = pd.read_csv('data/main_test_1')
    train_set.drop(columns=['has_collection', 'popularity', 'budget', 'language', 'runtime', 'revenue', 'vote_avg',
                            'vote_count', 'gen_one', 'gen_two', 'gen_three', 'gen_four'], inplace=True)
    test_set.drop(columns=['has_collection', 'popularity', 'budget', 'language', 'runtime', 'revenue', 'vote_avg',
                           'vote_count', 'gen_one', 'gen_two', 'gen_three', 'gen_four'], inplace=True)
    train_set = train_set.values.tolist()
    test_set = test_set.values.tolist()
    return train_set, test_set


def _save_test_train():
    df = _generate_set('data/train.csv')
    test_set = df.sample(n=7800)
    df.drop(index=test_set.index, inplace=True)
    test_set.to_csv('data/test_set_1', index=False)
    df.to_csv('data/train_set_1', index=False)


def test_train_split():
    # train_df = pd.read_csv('data/train_set_1').sample(n=500).values.tolist()
    # test_df = pd.read_csv('data/test_set_1').sample(n=100).values.tolist()
    train_df = pd.read_csv('data/train_set_1').values.tolist()
    test_df = pd.read_csv('data/test_set_1').values.tolist()
    train_x = [row[:-1] for row in train_df]
    train_y = [row[-1] for row in train_df]
    test_x = [row[:-1] for row in test_df]
    test_y = [row[-1] for row in test_df]
    return train_x, train_y, test_x, test_y, train_df, test_df


def test_train_drop():
    train_df = pd.read_csv('data/train_set_1')
    test_df = pd.read_csv('data/test_set_1')
    train_df.drop(columns=['has_collection', 'popularity', 'budget', 'language', 'runtime', 'revenue', 'vote_avg',
                           'vote_count'], inplace=True)
    test_df.drop(columns=['has_collection', 'popularity', 'budget', 'language', 'runtime', 'revenue', 'vote_avg',
                          'vote_count', 'gen_one','gen_two','gen_three','gen_four'], inplace=True)
    train_df = train_df.values.tolist()
    test_df = test_df.values.tolist()
    test_y = [row[-1] for row in test_df]
    return test_y, train_df, test_df


def test_train_pca():
    train_x, train_y, test_x, test_y, train_df, test_df = test_train_split()
    pca = PCA(n_components=8)
    train_x = pca.fit_transform(train_x).tolist()
    test_x = pca.transform(test_x).tolist()
    train_set = [row + [train_y[idx]] for idx, row in enumerate(train_x)]
    test_set = [row + [test_y[idx]] for idx, row in enumerate(test_x)]
    return test_y, train_set, test_set

# 3 has 4
def save_pca():
    test_y, train_set, test_set = test_train_pca()
    train_df = pd.DataFrame(train_set, columns=['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'rate'])
    test_df = pd.DataFrame(test_set, columns=['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'rate'])
    train_df.to_csv('data/pca_var/train_pca_7.csv', index=False)
    test_df.to_csv('data/pca_var/test_pca_7.csv', index=False)


def load_test_train_pca(version='all', num=0):
    if version == 'short':
        train_df = pd.read_csv('data/train_pca_short.csv').values.tolist()
        test_df = pd.read_csv('data/test_pca_short.csv').values.tolist()
    elif version == 'all':
        train_pth = 'data/pca_var/train_pca_' + str(num) + '.csv'
        test_pth = 'data/pca_var/test_pca_' + str(num) + '.csv'
        train_df = pd.read_csv(train_pth).values.tolist()
        test_df = pd.read_csv(test_pth).values.tolist()
    else:
        return
    test_y = [row[-1] for row in test_df]
    return test_y, train_df, test_df


def create_mixed_pca():
    train_df = pd.read_csv('data/train_set_1')
    test_df = pd.read_csv('data/test_set_1')
    train_df.drop(columns=['has_collection', 'popularity', 'budget', 'language', 'runtime', 'revenue', 'vote_avg',
                           'vote_count'], inplace=True)
    test_df.drop(columns=['has_collection', 'popularity', 'budget', 'language', 'runtime', 'revenue', 'vote_avg',
                          'vote_count'], inplace=True)
    test_y = [row[-1] for row in test_df.values.tolist()]
    train_gen = [row[2:6] for row in train_df.values.tolist()]
    test_gen = [row[2:6] for row in test_df.values.tolist()]
    pca = PCA(n_components=1)
    pca.fit(train_gen)
    train_gen = pca.transform(train_gen).tolist()
    test_gen = pca.transform(test_gen).tolist()
    train_gen = [int(x[0]) for x in train_gen]
    test_gen = [int(x[0]) for x in test_gen]
    train_df.drop(columns=['gen_one','gen_two','gen_three','gen_four'], inplace=True)
    test_df.drop(columns=['gen_one', 'gen_two', 'gen_three', 'gen_four'], inplace=True)
    train_df.insert(2, 'gen_pca', train_gen)
    test_df.insert(2, 'gen_pca', test_gen)
    print(train_df.head(20))
    print(test_df.head(20))
    train_df.to_csv('data/pca_var/train_pca_13.csv')
    test_df.to_csv('data/pca_var/test_pca_13.csv')
    return test_y, train_df, test_df

