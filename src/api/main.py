from fastapi import FastAPI
from google.cloud import storage
import torch
import pandas as pd
import sys
import os
# sys.path.append("../models")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
# from myapp import SomeObject

from src.models import model

import os

app = FastAPI()

@app.get("/")
def read_root():
   return {"Hello": "World"}


## Get Model from cloud
def getModelFromCloud():
    storage_client = storage.Client()
    bucket = storage_client.bucket("movie-rec-model-checkpoints")
    blob = bucket.blob("checkpoint.pt")

    # contents = blob.download_as_bytes()
    blob.download_to_filename("checkpoint.pt")
    net = model.load_checkpoint("checkpoint.pt")

    # print(net)
    return net
    # model = torch.hub.load('.', 'custom', 'yourmodel.pt', source='local')


## Get Processed test data from cloud
def gettest_dataFromCloud():
    storage_client = storage.Client()
    bucket = storage_client.bucket("movies-mlops-clean")
    blob = bucket.blob("test.pt")
    # contents = blob.download_as_bytes()
    blob.download_to_filename("test.pt")
    # test_data = torch.load("./data/processed/data.pt")
    test_data = torch.load("test.pt")
    # print(test_data)
    return test_data


## Get Movies raw data from cloud
def getMoviesFromCloud():
    gs_path = 'https://storage.googleapis.com/movies-mlops/movies_metadata.csv'
    movie_metadata = pd.read_csv(gs_path, usecols=['original_title'])
    movie_metadata.reset_index(inplace=True)
    movie_metadata = movie_metadata.rename(columns = {'index': 'customId'})
    movie_metadata['customId'] = movie_metadata['customId'] + 1

    return movie_metadata


def mapMovieIdToTitle(movie_metadata, id):
    movieTitle = movie_metadata[movie_metadata['customId'] == id]['original_title'].values[0]
    return movieTitle


def getuserPrediction(imported_model, userId, total_movies, movie_metadata, data):
    user_row = torch.tensor([userId] * total_movies)
    all_movie_ids = torch.arange(total_movies)
    edge_label_index = torch.stack([user_row, all_movie_ids], dim=0)
    pred = imported_model(data.x_dict, data.edge_index_dict, edge_label_index)
    print("before", pred)
    pred = pred.clamp(min=0.0, max=5.0)
    print("clam", pred)
    # we will only select movies for the user where the predicting rating is >3
    rec_movie_ids = (pred > 4).nonzero(as_tuple=True)
    print("after", pred)

    top_ten_recs = [rec_movies for rec_movies in rec_movie_ids[0].tolist()[:10]]

    top_ten_rec_titles = []
    for movieId in top_ten_recs:
        top_ten_rec_titles.append(mapMovieIdToTitle(movie_metadata, movieId))

    return top_ten_recs, top_ten_rec_titles


imported_model = getModelFromCloud()
test_data = gettest_dataFromCloud()
movie_metadata = getMoviesFromCloud()


@app.get("/predict/{user_id}")
def get_movies_prediction(user_id: int):
   total_movies = 9025
   top_ten_recs , top_ten_rec_titles= getuserPrediction(imported_model, user_id, total_movies, movie_metadata, test_data)
   
   print("predicting for", user_id)
   print(top_ten_recs)
   return {"top_ten_recs": top_ten_rec_titles}
