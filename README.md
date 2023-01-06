project_mlops
==============================

DTU MLOps Project

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


## Project Description

This repository contains the project work carried out in the MLOps course taught at DTU, in a group of 5 consisting of: 

Name  | Student number
------------- | -------------
Alexandra Polymenopoulou  | s212558
Bogdan Capsa  |   s210172
Jakob Fahl | s184419
Melina Siskou | s213158
Thomas Spyrou | s213161

#### Goal

The purpose of this project is to build a movie Recommendation System. In this project we will maninly focus on the pipeline of the system rather than the model itself.

#### Framework

We will be using Pytorch Geometric a library built upon PyTorch to easily write and train Graph Neural Networks (GNNs) for a wide range of applications related to structured data. Inference is performed using ONNX Runtime and fastai. Utilities are provided by Hydra. 

#### Data 

The Data is obtained using ArangoDB, containing 45463 records and 24 features. We are going to use the sampled version of The Movies Dataset. The dataset contains 3 csv files:

* **movies_metadata.csv**: Contains information on 45,000 movies featured in the Full MovieLens dataset. Features include posters, backdrops, budget, revenue, release dates, languages, production countries and companies.
* **links_small.csv**: Contains the TMDB and IMDB IDs of a small subset of 9,000 movies of the Full Dataset.
* **ratings_small.csv**: subset of 100,000 ratings from 700 users on 9,000 movies.

#### Model

This Graph Neural Network model is designed to predict the movies that a user has not watched yet, based on the links between the user and other movies.

Credits: https://medium.com/arangodb/integrate-arangodb-with-pytorch-geometric-to-build-recommendation-systems-dd69db688465
