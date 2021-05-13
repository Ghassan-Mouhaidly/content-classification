# Inpired by the work of https://github.com/rodrigobressan
# Modularity for any dataset/number of outputs coming soon !

import cv2
import numpy as np

from tensorflow.keras.utils import to_categorical


class DataGenerator(object):
    """
    """
    def __init__(self, df, dataset_dict, train_test_split, train_val_split, img_size):
        """
        """
        self.df = df
        self.dataset_dict = dataset_dict

        self.TRAIN_TEST_SPLIT = train_test_split
        self.TRAIN_VAL_SPLIT = train_val_split
        self.IMG_RES = img_size

        self.max_rating = self.df['Rating'].max()
        self.max_year = self.df['Year'].max()

        self.df['Genre'] = self.df['Genre'].map(lambda genre: self.dataset_dict['Genre_alias'][genre])
        
    def split_dataset(self):
        """
        """
        p = np.random.permutation(len(self.df))
        train_up_to = int(len(self.df) * self.TRAIN_TEST_SPLIT)

        train_idx = p[:train_up_to]
        test_idx = p[train_up_to:]   

        valid_idx = train_idx[int(train_up_to * self.TRAIN_VAL_SPLIT):]
        train_idx = train_idx[:int(train_up_to * self.TRAIN_VAL_SPLIT)]
        
        return train_idx, valid_idx, test_idx
    
    def _process_image(self, img_path):
        """
        """
        img = cv2.imread(img_path, 1)
        img = cv2.resize(img, self.IMG_RES[:2])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img) / 255.0
        
        return img
        
    def generate_images(self, image_idx, batch_size, is_training):
        """
        """
        images, genres, ratings, years = [], [], [], []

        while True:
            for idx in image_idx:
                movie = self.df.iloc[idx]
                
                genre = movie['Genre']
                rating = movie['Rating']
                year = movie['Year']
                image_path = movie['Poster_path']
                
                try:
                    img = self._process_image(image_path)
                except:
                    print("\n* * * * * Image could not be loaded - skipping * * * * *\n")
                    continue
                
                genres.append(to_categorical(genre, 8))
                ratings.append(rating / self.max_rating)
                years.append(year / self.max_year)
                images.append(img)
                
                if len(images) >= batch_size:
                    yield np.array(images), [np.array(genres), np.array(ratings), np.array(years)]

                    images, genres, ratings, years = [], [], [], []
                    
            if not is_training:
                break
