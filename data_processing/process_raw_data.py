import glob
import pandas as pd
import json
import os


def handle_exception(year, movie_idx):
    """
    """
    print('* * * * * * * * * * * * ')
    print(f'Movie {movie_idx} in year {year} has meta_data missing - skipping')
    print('* * * * * * * * * * * * ')

def process_raw(df, img_extention, poster_path, meta_data_path):
    """
    """
    years_path = glob.glob(poster_path + '*')

    for year_path in years_path:

        year = year_path.split('/')[-1]
        movies_path = glob.glob(year_path + '/*')
        year_movies = [movie.split('/')[-1] for movie in movies_path]
        
        for movie in year_movies:
            data_path = os.path.join(meta_data_path, year, movie, movie + '.json')
            movie_poster_path = os.path.join(poster_path, year, movie, movie + img_extention)

            f = open(data_path)
            data = json.load(f)

            try:
                genre = data['genre']

                if isinstance(genre, list):
                    genre = genre[0]

                df = df.append({'Index': str(movie), 'Name': str(data['name']), 'Year': int(year), 'Genre': str(genre), 
                            'Rating': float(data['aggregateRating']['ratingValue']), 
                            'Poster_path': str(movie_poster_path)}, 
                            ignore_index = True)
            except: 
                handle_exception(year, movie)
            
            f.close()

    return df
    

img_extention = '.jpg'
poster_path = "/home/robotics/Documents/data/Posters/"
meta_data_path = "/home/robotics/Documents/data/meta_data/"
df = pd.DataFrame(columns = ['Index', 'Name', 'Year', 'Genre', 'Rating', 'Poster_path'])

processed_df = process_raw(df, img_extention, poster_path, meta_data_path)

processed_df.to_pickle(os.path.join('/home/robotics/content-classification', 'data_processing', 'processed_data.pkl'))
