from userModel import User

test_movies = [{'features': [2, 5, 6], 'rate': 1},
               {'features': [3, 4, 1], 'rate': 3},
               {'features': [1, 3, 5], 'rate': 4},
               {'features': [5, 2, 3], 'rate': 5},
               {'features': [7, 1, 1], 'rate': 3},
               {'features': [3, 8, 2], 'rate': 2},
               {'features': [9, 1, 9], 'rate': 5},
               {'features': [3, 7, 0], 'rate': 2},
               {'features': [3, 3, 3], 'rate': 1},
               {'features': [2, 7, 5], 'rate': 1}]
test_user = User(3)
for t_mov in test_movies:
    test_user.movies.append(t_mov)

print(test_user.get_rate_for_new_movie([9, 1, 9], 3))
