from classifier import classify_val
from dataextractor import movie_feat_cols
import multiprocessing as mlpt
import copy


def select_best_feature(h_params):
    best = {'first': 0, 'second': 0, 'params': h_params, 'feat': 'none'}
    cols = copy.deepcopy(movie_feat_cols)
    cols.remove('has_collection')
    cols.remove('language')
    cols.remove('gen_two')
    cols.remove('vote_count')
    cols.remove('vote_avg')
    cols.remove('gen_three')
    cols.remove('popularity')
    cols.remove('gen_four')

    # for feat in cols:
    f, s = classify_val(['language', 'gen_two', 'vote_count', 'vote_avg', 'gen_three', 'popularity', 'gen_four'], h_params)
    if s > best['second']:
        best['first'] = f
        best['second'] = s
        best['feat'] = 'feat'

    print(best)


def search_multi(n_threads: int):
    h_params_set = [
        {'max_depth': 3, 'min_size': 55, 'method': 'gini'},
        {'max_depth': 5, 'min_size': 55, 'method': 'gini'},
        {'max_depth': 7, 'min_size': 55, 'method': 'gini'},
        {'max_depth': 10, 'min_size': 55, 'method': 'gini'},
        {'max_depth': 13, 'min_size': 55, 'method': 'gini'},
        {'max_depth': 16, 'min_size': 55, 'method': 'gini'},
        {'max_depth': 3, 'min_size': 55, 'method': 'entropy'},
        {'max_depth': 5, 'min_size': 55, 'method': 'entropy'},
        {'max_depth': 7, 'min_size': 55, 'method': 'entropy'},
        {'max_depth': 10, 'min_size': 55, 'method': 'entropy'},
        {'max_depth': 13, 'min_size': 55, 'method': 'entropy'},
        {'max_depth': 16, 'min_size': 55, 'method': 'entropy'}
    ]
    processes = [mlpt.Process(target=select_best_feature, args=(h_params_set[i],)) for i in range(n_threads)]
    for i in range(n_threads):
        processes[i].start()
    for i in range(n_threads):
        processes[i].join()


if __name__ == "__main__":
    search_multi(12)
