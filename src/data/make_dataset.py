
# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import oasis
from arango import ArangoClient
import pandas as pd
import numpy as np
from tqdm import tqdm
import yaml
import torch
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.transforms import RandomLinkSplit, ToUndirected
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData
import itertools
import subprocess


def remove_movies(m_id):
    no_metadata = []
    for idx in range(len(m_id)):
        tmdb_id = id_map.loc[id_map['movieId'] == m_id[idx]]

        if tmdb_id.size == 0:
            no_metadata.append(m_id[idx])
            #print('No Meta data information at:', m_id[idx])
    return no_metadata

def node_mappings(path, index_col):
    df = pd.read_csv(path, index_col=index_col)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    return mapping

def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan

def populate_user_collection(total_users):
    batch = []
    BATCH_SIZE = 50
    batch_idx = 1
    index = 0
    user_ids = list(user_mapping.keys())
    user_collection = movie_rec_db["Users"]
    for idx in tqdm(range(total_users)):
        insert_doc = {}

        insert_doc["_id"] = "Users/" + str(user_mapping[user_ids[idx]])
        insert_doc["original_id"] = str(user_ids[idx])

        batch.append(insert_doc)
        index +=1
        last_record = (idx == (total_users - 1))
        if index % BATCH_SIZE == 0:
            print("Inserting batch %d" % (batch_idx))
            batch_idx += 1
            user_collection.import_bulk(batch)
            batch = []
        if last_record and len(batch) > 0:
            print("Inserting batch the last batch!")
            user_collection.import_bulk(batch)


def create_ratings_graph(user_id, movie_id, ratings):
    batch = []
    BATCH_SIZE = 100
    batch_idx = 1
    index = 0
    edge_collection = movie_rec_db["Ratings"]
    for idx in tqdm(range(ratings.shape[0])):

        # removing edges (movies) with ltdata
        if movie_id[idx] in no_metadata:
            print('Removing edges with no metadata', movie_id[idx])

        else:
            insert_doc = {}
            insert_doc = {"_id":    "Ratings" + "/" + 'user-' + str(user_mapping[user_id[idx]]) + "-r-" + "movie-" + str(movie_mappings[movie_id[idx]]),
                          "_from":  ("Users" + "/" + str(user_mapping[user_id[idx]])),
                          "_to":    ("Movie" + "/" + str(movie_mappings[movie_id[idx]])),
                          "_rating": float(ratings[idx])}

            batch.append(insert_doc)
            index += 1
            last_record = (idx == (ratings.shape[0] - 1))

            if index % BATCH_SIZE == 0:
                #print("Inserting batch %d" % (batch_idx))
                batch_idx += 1
                edge_collection.import_bulk(batch)
                batch = []
            if last_record and len(batch) > 0:
                print("Inserting batch the last batch!")
                edge_collection.import_bulk(batch)
def create_pyg_edges(rating_docs):
    src = []
    dst = []
    ratings = []
    for doc in rating_docs:
        _from = int(doc['_from'].split('/')[1])
        _to   = int(doc['_to'].split('/')[1])

        src.append(_from)
        dst.append(_to)
        ratings.append(int(doc['_rating']))

    edge_index = torch.tensor([src, dst])
    edge_attr = torch.tensor(ratings)

    return edge_index, edge_attr

def SequenceEncoder(movie_docs , model_name=None):
    movie_titles = [doc['movie_title'] for doc in movie_docs]
    model = SentenceTransformer(model_name, device=device)
    title_embeddings = model.encode(movie_titles, show_progress_bar=True,
                              convert_to_tensor=True, device=device)

    return title_embeddings

def GenresEncoder(movie_docs):
    gen = []
    #sep = '|'
    for doc in movie_docs:
        gen.append(doc['genres'])
        #genre = doc['movie_genres']
        #gen.append(genre.split(sep))

    # getting unique genres
    unique_gen = set(list(itertools.chain(*gen)))
    print("Number of unqiue genres we have:", unique_gen)

    mapping = {g: i for i, g in enumerate(unique_gen)}
    x = torch.zeros(len(gen), len(mapping))
    for i, m_gen in enumerate(gen):
        for genre in m_gen:
            x[i, mapping[genre]] = 1
    return x.to(device)

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())


    ## Loading the data present in csv files to ArangoDB
    raw_data_dir = './data/raw'
    processed_data_dir = './data/processed'
    # Load metadata
    df = pd.read_csv (raw_data_dir + '/movies_metadata.csv')
    # on these rows metadata information is missing
    df = df.drop([19730, 29503, 35587])
    # Load links
    # Sampled from links.csv file
    links_small = pd.read_csv(raw_data_dir + '/links_small.csv')
    # Selecting tmdbId coloumn from links_small file
    links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
    df['id'] = df['id'].astype('int')
    sampled_md = df[df['id'].isin(links_small)]
    sampled_md['tagline'] = sampled_md['tagline'].fillna('')
    sampled_md['description'] = sampled_md['overview'] + sampled_md['tagline']
    sampled_md['description'] = sampled_md['description'].fillna('')
    sampled_md = sampled_md.reset_index()
    indices = pd.Series(sampled_md.index, index=sampled_md['title'])
    ind_gen = pd.Series(sampled_md.index, index=sampled_md['genres'])
    # Load ratings
    ratings_path = raw_data_dir + '/ratings_small.csv'
    ratings_df = pd.read_csv(ratings_path)
    user_mapping = node_mappings(ratings_path, index_col='userId')
    movie_mapping = node_mappings(ratings_path, index_col='movieId')
    m_id = ratings_df['movieId'].tolist()
    m_id = list(dict.fromkeys(m_id))
    id_map = pd.read_csv(raw_data_dir + '/links_small.csv')[['movieId', 'tmdbId']]
    id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
    id_map.columns = ['movieId', 'id']
    id_map = id_map.merge(sampled_md[['title', 'id']], on='id').set_index('title')
    indices_map = id_map.set_index('id')
    # remove ids which dont have meta data information
    no_metadata = remove_movies(m_id)
    for element in no_metadata:
        if element in m_id:
            m_id.remove(element)
    # create new movie_mapping dict with only m_ids having metadata information
    movie_mappings = {}
    for idx, m in enumerate(m_id):
        movie_mappings[m] = idx

    ## ArangoDB Setup
    # get temporary credentials for ArangoDB on cloud
    login = oasis.getTempCredentials(tutorialName="MovieRecommendations", credentialProvider="https://tutorials.arangodb.cloud:8529/_db/_system/tutorialDB/tutorialDB")
    # Connect to the temp database
    # Please note that we use the python-arango driver as it has better support for ArangoSearch
    movie_rec_db = oasis.connect_python_arango(login)

    # url to access the ArangoDB Web UI
    print("https://"+login["hostname"]+":"+str(login["port"]))
    print("Username: " + login["username"])
    print("Password: " + login["password"])
    print("Database: " + login["dbName"])

    ## Loading movies metadata into ArangoDB's Movie collection
    # create a new collection named "Movie" if it does not exist.
    # This returns an API wrapper for "Movie" collection.
    if not movie_rec_db.has_collection("Movie"):
        movie_rec_db.create_collection("Movie", replication_factor=3)
    batch = []
    BATCH_SIZE = 128
    batch_idx = 1
    index = 0
    movie_collection = movie_rec_db["Movie"]
    # loading movies metadata information into ArangoDB's Movie collection
    for idx in tqdm(range(len(m_id))):
        insert_doc = {}
        tmdb_id = id_map.loc[id_map['movieId'] == m_id[idx]]

        if tmdb_id.size == 0:
            print('No Meta data information at:', m_id[idx])
        else:
            tmdb_id = int(tmdb_id.iloc[:,1][0])
            emb_id = "Movie/" + str(movie_mappings[m_id[idx]])
            insert_doc["_id"] = emb_id
            m_meta = sampled_md.loc[sampled_md['id'] == tmdb_id]
            # adding movie metadata information
            m_title = m_meta.iloc[0]['title']
            m_poster = m_meta.iloc[0]['poster_path']
            m_description = m_meta.iloc[0]['description']
            m_language = m_meta.iloc[0]['original_language']
            m_genre = m_meta.iloc[0]['genres']
            m_genre = yaml.load(m_genre, Loader=yaml.BaseLoader)
            genres = [g['name'] for g in m_genre]

            insert_doc["movieId"] = m_id[idx]
            insert_doc["mapped_movieId"] = movie_mappings[m_id[idx]]
            insert_doc["tmdbId"] = tmdb_id
            insert_doc['movie_title'] = m_title

            insert_doc['description'] = m_description
            insert_doc['genres'] = genres
            insert_doc['language'] = m_language

            if str(m_poster) == "nan":
                insert_doc['poster_path'] = "No poster path available"
            else:
                insert_doc['poster_path'] = m_poster

            batch.append(insert_doc)
            index +=1
            last_record = (idx == (len(m_id) - 1))
            if index % BATCH_SIZE == 0:
                #print("Inserting batch %d" % (batch_idx))
                batch_idx += 1
                movie_collection.import_bulk(batch)
                batch = []
            if last_record and len(batch) > 0:
                print("Inserting batch the last batch!")
                movie_collection.import_bulk(batch)


    # Users has no side information
    total_users = np.unique(ratings_df[['userId']].values.flatten()).shape[0]
    print("Total number of Users:", total_users)

    ## Creating User Collection in ArangoDB
    # create a new collection named "Users" if it does not exist.
    # This returns an API wrapper for "Users" collection.
    if not movie_rec_db.has_collection("Users"):
        movie_rec_db.create_collection("Users", replication_factor=3)

    populate_user_collection(total_users)

    ## Creating Ratings (Edge) Collection
    # create a new collection named "Ratings" if it does not exist.
    # This returns an API wrapper for "Ratings" collection.
    if not movie_rec_db.has_collection("Ratings"):
        movie_rec_db.create_collection("Ratings", edge=True, replication_factor=3)

    # defining graph schema
    # create a new graph called movie_rating_graph in the temp database if it does not already exist.
    if not movie_rec_db.has_graph("movie_rating_graph"):
        movie_rec_db.create_graph('movie_rating_graph', smart=True)

    # This returns and API wrapper for the above created graphs
    movie_rating_graph = movie_rec_db.graph("movie_rating_graph")

    # Create a new vertex collection named "Users" if it does not exist.
    if not movie_rating_graph.has_vertex_collection("Users"):
        movie_rating_graph.vertex_collection("Users")

    # Create a new vertex collection named "Movie" if it does not exist.
    if not movie_rating_graph.has_vertex_collection("Movie"):
        movie_rating_graph.vertex_collection("Movie")

    #movie_rec_db._drop(["Users", "Movie", "Ratings"])

    print("-------------------------------------")

    # creating edge definitions named "Ratings. This creates any missing
    # collections and returns an API wrapper for "Ratings" edge collection.
    if not movie_rating_graph.has_edge_definition("Ratings"):
        Ratings = movie_rating_graph.create_edge_definition(
            edge_collection='Ratings',
            from_vertex_collections=['Users'],
            to_vertex_collections=['Movie']
        )

    user_id, movie_id, ratings = ratings_df[['userId']].values.flatten(), ratings_df[['movieId']].values.flatten() , ratings_df[['rating']].values.flatten()

    create_ratings_graph(user_id, movie_id, ratings)

    ## Converting the Graph present inside the ArangoDB into a PyTorch Geometric (PyG) data object
    # Get API wrappers for collections.
    users = movie_rec_db.collection('Users')
    movies = movie_rec_db.collection('Movie')
    ratings_graph = movie_rec_db.collection('Ratings')

    # graphs = movie_rec_db._list_graphs()
    # for graph in graphs:
    #     my_graph = movie_rec_db._graph(graph)
    #     print(my_graph["edgeDefinitions"])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load edges from Ratings collection in ArangoDB and export them to PyG data format.
    edge_index, edge_label = create_pyg_edges(movie_rec_db.aql.execute('FOR doc IN Ratings RETURN doc'))
    print(edge_index.shape)
    print(edge_label.shape)

    ## Load nodes from Ratings collection in ArangoDB and export them PyG data format.
    title_emb = SequenceEncoder(movie_rec_db.aql.execute('FOR doc IN Movie RETURN doc'), model_name='all-MiniLM-L6-v2')
    encoded_genres = GenresEncoder(movie_rec_db.aql.execute('FOR doc IN Movie RETURN doc'))
    print('Title Embeddings shape:', title_emb.shape)
    print("Encoded Genres shape:", encoded_genres.shape)

    # concat title and genres features of movies
    movie_x = torch.cat((title_emb, encoded_genres), dim=-1)
    print("Shape of the concatenated features:", movie_x.shape)

    ## Creating PyG Heterogeneous Graph
    data = HeteroData()
    data['user'].num_nodes = len(users)  # Users do not have any features.
    data['movie'].x = movie_x
    data['user', 'rates', 'movie'].edge_index = edge_index
    data['user', 'rates', 'movie'].edge_label = edge_label

    # Add user node features for message passing:
    data['user'].x = torch.eye(data['user'].num_nodes, device=device)
    del data['user'].num_nodes

    # Add a reverse ('movie', 'rev_rates', 'user') relation for message passing.
    data = ToUndirected()(data)
    del data['movie', 'rev_rates', 'user'].edge_label  # Remove "reverse" label.

    data = data.to(device)

    # Perform a link-level split into training, validation, and test edges.
    train_data, val_data, test_data = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        neg_sampling_ratio=0.0,
        edge_types=[('user', 'rates', 'movie')],
        rev_edge_types=[('movie', 'rev_rates', 'user')],
    )(data)

    print('Train data:', train_data)
    print('Val data:', val_data)
    print('Test data', test_data)

    torch.save(train_data, './data/processed/train.pt')
    torch.save(val_data, './data/processed/val.pt')
    torch.save(test_data, './data/processed/test.pt')

    print('Pushing processed data to dvc...')
    subprocess.run(['dvc','push','./data/processed/*.pt','-r','processed'])

    main()
