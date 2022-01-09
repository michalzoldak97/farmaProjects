class User:

    def __init__(self, user_id: int, movies=None):
        if movies is None:
            movies = []
        self.user_id = user_id
        self.movies = movies

    def has_a_movie(self, idx: int):
        movies_idx = [mov[0] for mov in self.movies]
        return idx in movies_idx

    def get_movie(self, idx: int):
        for mov in self.movies:
            if idx == mov[0]:
                return mov

