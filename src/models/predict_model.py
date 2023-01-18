from google.cloud import storage
import torch
import pandas as pd

## Get Model from cloud
def getModelFromCloud():
    storage_client = storage.Client()
    bucket = storage_client.bucket("movie-rec-model-checkpoints")
    blob = bucket.blob("checkpoint.pt")
    # contents = blob.download_as_bytes()
    blob.download_to_filename("checkpoint.pt")
    model = torch.load("checkpoint.pt")
    print(model)
    return model
    # model = torch.hub.load('.', 'custom', 'yourmodel.pt', source='local')


## Get Processed test data from cloud
def getTestDataFromCloud():
    storage_client = storage.Client()
    bucket = storage_client.bucket("movies-mlops-clean")
    blob = bucket.blob("test.pt")
    # contents = blob.download_as_bytes()
    blob.download_to_filename("test.pt")
    test_data = torch.load("test.pt")
    print(test_data)
    return test_data


## Get Movies raw data from cloud
def getMoviesFromCloud():
    gs_path = 'https://storage.googleapis.com/movies-mlops/movies_metadata.csv'
    movie_metadata = pd.read_csv(gs_path, usecols=['original_title'])
    movie_metadata.reset_index(inplace=True)
    movie_metadata = movie_metadata.rename(columns = {'index': 'customId'})
    movie_metadata['customId'] = movie_metadata['customId'] + 1
    return movie_metadata


def mapMovieIdToTitle(movieMetadata, id):
    movieTitle = movieMetadata[movieMetadata['customId'] == id]['original_title'].values[0]
    return movieTitle
    

def getuserPrediction(userId, total_movies, movieMetadata, data):
    user_row = torch.tensor([userId] * total_movies)
    all_movie_ids = torch.arange(total_movies)
    edge_label_index = torch.stack([user_row, all_movie_ids], dim=0)
    pred = model(data.x_dict, data.edge_index_dict, edge_label_index)
    pred = pred.clamp(min=0, max=5)
    # we will only select movies for the user where the predicting rating is =5
    rec_movie_ids = (pred == 5).nonzero(as_tuple=True)
    top_ten_recs = [rec_movies for rec_movies in rec_movie_ids[0].tolist()[:10]] 
    
    top_ten_rec_titles = []
    for movieId in top_ten_recs:
        movieTitle = top_ten_rec_titles.append(mapMovieIdToTitle(movieMetadata, movieId))
        
    return top_ten_recs, top_ten_rec_titles


if __name__ == "__main__":
    model = getModelFromCloud()
    testData = getTestDataFromCloud()
    movieMetadata = getMoviesFromCloud()

    # todo: fix this
    # total_movies = len(movies)
    total_movies = 9025
    userId = 0
    getuserPrediction(userId, total_movies, movieMetadata, testData)





