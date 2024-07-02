import numpy as np
import ast
import warnings
import pandas as pd
from collections import Counter
import json
import os


class LoadDataset:

    def  __init__(self, filename):
        self.__filename = filename

    def load(self):
        df =  pd.read_csv(self.__filename, low_memory=False)
        if df.shape[0] !=0:
            print(f"Dataset loaded: {self.__filename}")
            print(f"Rows: {df.shape[0]}\nColumns: {df.shape[1]}")
            return df
        else:
            print('No data loaded')

class DatasetOperations:

    def __init__(self, df):
        self.df = df
    
    def data_preprocessing(self):
        """
        Data preprocessing/cleaning procedure for 3 rows that seem to be split into 2 rows.

        Returns:
            pandas DataFrame: Dataframe after mini preprocessing procedure
        """
        try:
            for i in [19729,29502,35586]:
                self.df.iloc[i,9] = self.df.iloc[i,9] + self.df.iloc[i+1,0]
                self.df.iloc[i,10] = float(self.df.iloc[i+1,1])                 # popularity
                self.df.iloc[i,11] = str(self.df.iloc[i+1,2])                   # poster_path
                self.df.loc[[i],'production_companies'] = pd.Series([self.df.iloc[i+1,3]], index = self.df.index[[i]])  # production_companies
                self.df.loc[[i],'production_countries'] = pd.Series([self.df.iloc[i+1,4]], index = self.df.index[[i]])  # production_countries
                self.df.iloc[i,14] = self.df.iloc[i+1,5]                        # release_date
                self.df.iloc[i,15] = float(self.df.iloc[i+1,6])                 # revenue
                self.df.iloc[i,16] = float(self.df.iloc[i+1,7])                 # runtime
                self.df.iloc[i,17] = ast.literal_eval(self.df.iloc[i+1,8])      # spoken_languages
                self.df.iloc[i,18] = str(self.df.iloc[i+1,9])                   # status
                self.df.iloc[i,19] = np.nan                                     # tagline
                self.df.iloc[i,20] = str(self.df.iloc[i+1,11])                  # title
                self.df.iloc[i,21] = np.nan                                     # video
                self.df.iloc[i,22] = float(self.df.iloc[i+1,13])                # vote_average
                self.df.iloc[i,23] = float(self.df.iloc[i+1,14])                # vote_count
            self.df.drop([19730,29503,35587], inplace = True)
            print("Data Pre-Processing completed")
            return self.df
        except:
            print("Failed pre-processing")

    def unique_movies_number(self):
        """
        Counts unique movies in the dataset.

        Returns:
            integer: Amount of unique movies
        """
        
        unique_movies_nr = self.df[['original_title','release_date']].value_counts().count()
        print(f"Number of the unique movies: {unique_movies_nr}")
    
    def average_movie_rating(self):
        """
        Calculates average movie rating.
        
        Returns:
            float: Average movies rating
        """

        average_movie_rating = round(self.df.vote_average.mean(),2)
        print(f"Average rating of all the movies: {average_movie_rating}")

    def top_five_rated(self):
        """
        Calculates the 5 top-rated movies with at least 10K votes based on vote average.

        Returns:
            list: 5 top-rated movies
        """
        top_five = self.df.query('vote_count>9999.0').nlargest(5, 'vote_average')['original_title'].to_list()
        print("Top 5 highest rated movies:", *top_five, sep = '\n')

    def releases_per_year(self):
        """
        Finds how many movies are released each year.
        
        Returns:
            list: Year-release amount pairs
        """
        warnings.simplefilter(action='ignore', category=UserWarning)
        year_value_count = self.df[self.df.status=='Released'][~self.df['release_date'].isna()].release_date.str[:4].value_counts().sort_index()
        print("\n".join(f"{year}: {year_count}" for year, year_count in zip(year_value_count.index.to_list(),  year_value_count.to_list())))

    def movies_per_genre(self):
        """
        Finds how many movies each genre has.

        Returns:
            dict: key=genre, value=amount of movies per genre
        """
        genres_list = [value['name'] for x in self.df.genres.to_list() for value in eval(x)]
        genre_counter = Counter( genres_list )
        genre_counts = sorted(genre_counter.items(), key = lambda i: i[0])
        print("\n".join(f"{genre}: {genre_count}" for genre, genre_count in genre_counts ))

    def save_to_json(self):
        """
        Saves pandas Dataframe in a json file.

        Returns:
            json file
        """
        result = self.df.to_dict('records')
        filepath = "data/"
        filename = "movies_metadata.json"
        if not os.path.exists(filepath+filename):
            with open(filepath+filename, "w") as outfile: 
                json.dump(result, outfile)
        else:
            print('JSON file already exists!')
        
m_df = LoadDataset('data/movies_metadata.csv').load()
clean_df = DatasetOperations(m_df).data_preprocessing()
DatasetOperations(clean_df).unique_movies_number()
DatasetOperations(clean_df).average_movie_rating()
DatasetOperations(clean_df).top_five_rated()
DatasetOperations(clean_df).releases_per_year()
DatasetOperations(clean_df).movies_per_genre()
DatasetOperations(clean_df).save_to_json()
