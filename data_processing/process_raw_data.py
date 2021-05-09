import glob
import pandas as pd
import json
import pickle
import os

def handle_exception(year, movie_idx):
    """
    """
    print('* * * * * * * * * * * * ')
    print(f'Movie {movie_idx} in year {year} has meta_data missing - skipping')
    print('* * * * * * * * * * * * ')


df = pd.DataFrame(columns = ['Movie index', 'Movie name', 'Year', 'Genres', 'Rating', 'Poster path'])

img_extention = 'jpg'
poster_path = "/home/robotics/Documents/data/Posters/"
meta_data_path = "/home/robotics/Documents/data/meta_data/"

years_path = glob.glob(poster_path + '*')

for year_path in years_path:
    year = year_path.split('/')[-1]

    movies_path = glob.glob(year_path + '/*')
    year_movies = [movie.split('/')[-1] for movie in movies_path]
    
    for movie in year_movies:
        data_path = os.path.join(meta_data_path, year, movie, movie + '.json')
        poster_path = os.path.join(poster_path, year, movie, movie + img_extention)

        f = open(data_path)
        data = json.load(f)
        try:
            df.append({'Movie index' : movie, 'Movie name' : data['name'], 'Year' : year, 'Genres': data['genre'], 
                        'Rating': data['aggregateRating']['ratingValue'], 
                        'Poster path': poster_path}, 
                        ignore_index = True)
        except: 
            handle_exception(year, movie)
        
        f.close()

df.to_pickle(os.path.join('/home/robotics/content-classification', 'full_content_data.pkl'))
