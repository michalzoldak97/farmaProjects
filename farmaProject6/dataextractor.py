import requests
import json
import csv
import os
import sys

api_key = '450b3049deaa6b6f55fa1a8f8880e8b1'
movie_id = '464052'

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
        budget = dataset['budget']
        genres_all = dataset['genres']
        genres = ''
        for gen in genres_all:
            genres += gen['name'] + '; '
    except:
        print("Error gen")

    # sys.exit()

    result = [dataset['original_title'],collection_name, budget, genres]
    csvwriter.writerow(result)
    print (result)
    csvFile.close()

csvFile = open('movie_collection_data.csv','w')
csvwriter = csv.writer(csvFile)
csvwriter.writerow(['Movie_name','Collection_name', 'Budget', 'Genres'])
csvFile.close()
text = get_data(api_key, movie_id)
if text == "error":
    print(error)
else:
    write_file('movie_collection_data.csv', text)
