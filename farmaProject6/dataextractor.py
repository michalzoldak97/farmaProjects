import requests
import json
import csv
import pandas as pd
import sys

api_key = '450b3049deaa6b6f55fa1a8f8880e8b1'
movie_id = '464052'

movie_ids = pd.read_csv('movie.csv', sep=';')['m_idx']

movie_ids = [idx for idx in movie_ids]


def get_data(a_key, mv_id):
    query = 'https://api.themoviedb.org/3/movie/'+mv_id+'?api_key='+a_key
    res =  requests.get(query)
    if res.status_code==200: 
        array = res.json()
        text = json.dumps(array)
        return (text)
    else:
        return ("error")


def write_file(filename, text):
    dataset = json.loads(text)
    csvFile = open(filename,'a')
    csvwriter = csv.writer(csvFile)
    try:
        collection_name = dataset['belongs_to_collection']['name']
    except:
        collection_name = None
    try:
        adult = dataset['adult']
        budget = dataset['budget']
        genres_all = dataset['genres']
        genres = ''
        for gen in genres_all:
            genres += gen['name'] + '; '
        original_language = dataset['original_language']
        popularity = dataset['popularity']
        revenue = dataset['revenue']
        runtime = dataset['runtime']
        vote_average = dataset['vote_average']
        vote_count = dataset['vote_count']

    except:
        print("Error gen")


    result = [dataset['original_title'], adult, collection_name, budget, genres, original_language, popularity, revenue, runtime, vote_average, vote_count]
    csvwriter.writerow(result)
    print (result)
    csvFile.close()


csvFile = open('movie_collection_data.csv','w')
csvwriter = csv.writer(csvFile)
csvwriter.writerow(['Movie_name','Adult', 'Collection_name', 'Budget', 'Genres', 'Language', 'Popularity', 'Revenue', 'Runtime', 'Vote AVG', 'Vote CNT'])
csvFile.close()
for idx in movie_ids:
    text = get_data(api_key, str(idx))
    if text == "error":
        print("err")
    else:
        write_file('movie_collection_data.csv', text)
