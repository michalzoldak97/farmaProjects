import pandas as pd

mov_data = pd.read_csv('movie_collection_data_3.csv')


def index_names():
    for i, _ in enumerate(mov_data['Movie_name']):
        mov_data.at[i, 'Movie_name'] = str(i + 1)
    mov_data.drop(columns=['Adult'], inplace=True)
    mov_data.to_csv('movie_collection_data_1.csv', index=False)


def sec_data():
    for i, _ in enumerate(mov_data['Collection_name']):
        if pd.isna(mov_data.at[i, 'Collection_name']):
            mov_data.at[i, 'Collection_name'] = 0
        else:
            mov_data.at[i, 'Collection_name'] = 1
    mov_data.to_csv('movie_collection_data_1.csv', index=False)


def sec_unig_data():
    for i, col in enumerate(mov_data.Genres):
        temp_res = 0
        for el in col.split(';'):
            els = el.strip()
            if els == "Drama":
                temp_res += .1
            if els == "Science Fiction":
                temp_res += 3
            if els == "Mystery":
                temp_res += 5
            if els == "Adventure":
                temp_res += 2    
            if els == "Comedy":
                temp_res += .001  
            if els == "Romance":
                temp_res += .0001
            if els == "Horror":
                temp_res += 100
            if els == "War":
                temp_res += 90
            if els == "History":
                temp_res += 0.5
            if els == "Music":
                temp_res += 0.3
            if els == "Thriller":
                temp_res += 80
            if els == "Crime":
                temp_res += 80
            if els == "Family":
                temp_res += 1
            if els == "Action":
                temp_res += 10
            if els == "Animation":
                temp_res += .1
            if els == "Fantasy":
                temp_res += 4
            if els == "Western":
                temp_res += 20
        mov_data.at[i, 'Genres'] = temp_res

    for i, lng in enumerate(mov_data.Language):
        if lng == "en":
            mov_data.at[i, 'Language'] = 100
        elif lng == "hi":
            mov_data.at[i, 'Language'] = 0
        elif lng == "es":
            mov_data.at[i, 'Language'] = 16
        elif lng == "pt":
            mov_data.at[i, 'Language'] = 15
        elif lng == "sv":
            mov_data.at[i, 'Language'] = 19
        elif lng == "da":
            mov_data.at[i, 'Language'] = 6
        elif lng == "it":
            mov_data.at[i, 'Language'] = 20
        elif lng == "fr":
            mov_data.at[i, 'Language'] = 25
        elif lng == "cn":
            mov_data.at[i, 'Language'] = -1
        elif lng == "ja":
            mov_data.at[i, 'Language'] = 7
        elif lng == "de":
            mov_data.at[i, 'Language'] = 28
             
    print(mov_data.head(20))
    mov_data.to_csv('movie_collection_data_2.csv', index=False)


def vec_add(v1, v2):
    return [x + y for x, y in zip(v1, v2)]


def vectorize():
    for i, lng in enumerate(mov_data.Language):
        if lng == "en":
            mov_data.at[i, 'Language'] = 100
        elif lng == "hi":
            mov_data.at[i, 'Language'] = 0
        elif lng == "es":
            mov_data.at[i, 'Language'] = 16
        elif lng == "pt":
            mov_data.at[i, 'Language'] = 15
        elif lng == "sv":
            mov_data.at[i, 'Language'] = 19
        elif lng == "da":
            mov_data.at[i, 'Language'] = 6
        elif lng == "it":
            mov_data.at[i, 'Language'] = 20
        elif lng == "fr":
            mov_data.at[i, 'Language'] = 25
        elif lng == "cn":
            mov_data.at[i, 'Language'] = -1
        elif lng == "ja":
            mov_data.at[i, 'Language'] = 7
        elif lng == "de":
            mov_data.at[i, 'Language'] = 28

    genre_dict = {
        'Drama': [7, 5, 5, 7],
        'Science Fiction': [5, 8, 7, 3],
        'Mystery': [6, 7, 7, 4],
        'Adventure': [4, 10, 5, 5],
        'Comedy': [1, 5, 3, 8],
        'Romance': [2, 5, 2, 10],
        'Horror': [10, 8, 2, 2],
        'War': [9, 9, 7, 3],
        'History': [3, 3, 10, 3],
        'Music': [2, 4, 5, 8],
        'Thriller': [7, 7, 6, 3],
        'Crime': [8, 7, 7, 3],
        'Family': [0, 8, 6, 7],
        'Action': [7, 10, 5, 5],
        'Animation': [2, 7, 6, 7],
        'Fantasy': [6, 8, 6, 6],
        'Western': [8, 8, 3, 6]
    }
    gen_one = []
    gen_two = []
    gen_three = []
    gen_four = []
    for i, col in enumerate(mov_data.Genres):
        vec = [0, 0, 0, 0]
        for el in col.split(';'):
            els = el.strip()
            for k in genre_dict.keys():
                if els == k:
                    vec = vec_add(vec, genre_dict[k])

        gen_one.append(vec[0])
        gen_two.append(vec[1])
        gen_three.append(vec[2])
        gen_four.append(vec[3])

    mov_data["gen_one"] = gen_one
    mov_data["gen_two"] = gen_two
    mov_data["gen_three"] = gen_three
    mov_data["gen_four"] = gen_four

    mov_data.drop(columns=["Genres"], inplace=True)
    print(mov_data.head(20))
    mov_data.to_csv('movie_collection_data_3.csv', index=False)


def scale_col():
    for i, col in enumerate(mov_data):
        if i == 1:
            mov_data[col] = mov_data[col] * 10
        elif i == 2:
            mov_data[col] = mov_data[col] * .000001
        elif i == 4:
            mov_data[col] = mov_data[col] * 10
        elif i == 5:
            mov_data[col] = mov_data[col] * .000001
        elif i == 6:
            mov_data[col] = mov_data[col] * .1
        elif i == 7:
            mov_data[col] = mov_data[col] * 100
        elif i == 8:
            mov_data[col] = mov_data[col] * .01

    print(mov_data.head())
    mov_data.to_csv('movie_collection_data_4.csv', index=False)


def build_train_set():
    train_df = pd.read_csv('train.csv', sep=';')
    train_df.user = pd.to_numeric(train_df.user, downcast='float')
    mov_feat_df = pd.read_csv('movie_collection_data_4.csv')
    train_df.insert(2, 'has_collection', 0.0)
    train_df.insert(3, 'budget', 0.0)
    train_df.insert(4, 'language', 0.0)
    train_df.insert(5, 'popularity', 0.0)
    train_df.insert(6, 'revenue', 0.0)
    train_df.insert(7, 'runtime', 0.0)
    train_df.insert(8, 'vote_avg', 0.0)
    train_df.insert(9, 'vote_count', 0.0)
    train_df.insert(10, 'gen_one', 0.0)
    train_df.insert(11, 'gen_two', 0.0)
    train_df.insert(12, 'gen_three', 0.0)
    train_df.insert(13, 'gen_four', 0.0)

    for idx in train_df.index:
        movie = train_df.at[idx, 'movie']
        for col in train_df:
            try:
                train_df.at[idx, col] = float(mov_feat_df.at[movie, col])
            except KeyError:
                continue

        train_df.at[idx, 'user'] = ((float(train_df.at[idx, 'user']) * 1.0) / 1816.0) * 100.0

    train_df.drop(columns=['idx'], inplace=True)

    train_df.to_csv('train_data_1.csv', index=False)

    print(train_df.head(20))
    print(len(train_df.user.unique()))


def show_coor():
    tr_df = pd.read_csv('train_data_1.csv')
    print(tr_df.corr()[['rate']])
    cnt = 0
    for x in tr_df.user.values:
        if x == 90.4185:
            cnt += 1
    print(cnt)


show_coor()
