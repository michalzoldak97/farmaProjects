import warnings
from distances import euc_dist
# has id int
# has movies as a list of {features: list, rate: int}


class User:
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.movies = []

    def _get_most_common_rate(self, rates: list):
        rate_nums = {}
        for rate in rates:
            if rate in rate_nums.keys():
                rate_nums[rate] += 1
            else:
                rate_nums[rate] = 1

        return max(rate_nums, key=rate_nums.get)

    def _rank_by_similarity(self, new_movie_feats: list):
        # foreach user movie calc (distance, rate)
        similarity_ranking = [(euc_dist(new_movie_feats, user_movie['features']),  user_movie['rate']) for user_movie in
                              self.movies]
        similarity_ranking.sort(key=lambda tup: tup[0])
        return similarity_ranking

    def _classify_new_movie(self, new_movie_feats: list, top_similar: int):
        top_k_neighs = self._rank_by_similarity(new_movie_feats)
        top_k_rates = [top_k_neighs[i][1] for i in range(top_similar)]
        return self._get_most_common_rate(top_k_rates)

    def get_rate_for_new_movie(self, new_movie_feats: list, top_similar=5):
        my_movies_num = len(self.movies)

        if top_similar > my_movies_num > 0:
            warnings.warn('K is greater than user movies num, will return most common rate')
            top_similar = my_movies_num
        if my_movies_num < 1:
            raise ValueError('User has no movies')
        if top_similar < 1:
            raise ValueError('K can not be below 1')
        if len(new_movie_feats) < len(self.movies[0]['features']):
            raise ValueError('Invalid movie features format')

        return self._classify_new_movie(new_movie_feats, top_similar)
