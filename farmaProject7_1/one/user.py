from tree import build_tree, Leaf, Node, Condition


class User:
    def __init__(self, user_id: int, max_depth=10, min_size=10, method='gini'):
        self.user_id = user_id
        self.max_depth = max_depth
        self.min_size = min_size
        self.method = method
        self.movies = []
        self.user_tree = None

    def build_user_tree(self):
        if len(self.movies) < self.min_size:
            raise ValueError('Movies not assigned or size too high')

        data = []
        for i, mov in enumerate(self.movies):
            data.append(mov['features'])
            data[i].append(mov['rate'])

        self.user_tree = build_tree(data, 0, self.max_depth, self.min_size, self.method)

    def _classify_with_tree(self, row, node):
        if isinstance(node, Leaf):
            return node.rate
        if node.condition.is_con(row):
            return self._classify_with_tree(row, node.l_branch)
        else:
            return self._classify_with_tree(row, node.r_branch)

    def get_rate_for_new_movie(self, mov_feats: list):
        return self._classify_with_tree(mov_feats, self.user_tree)
