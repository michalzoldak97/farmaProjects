
class User:
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.movies = []

    def get_rate_for_new_movie(self, max_depth=10, min_size=10):
        pass