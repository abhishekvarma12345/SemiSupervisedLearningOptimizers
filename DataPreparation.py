import time
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import rbf_kernel
from common import read_config
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def prepare_dataset(path):
    config = read_config(path)
    testsize = config.get('testsize')
    max_iter = config.get('maxiterations')
    eps = config.get('epsilon2')
    randomseed = config.get('randomseed')

    df = pd.read_csv ("citrus.csv")
    X = df[['diameter', 'weight', 'red', 'green', 'blue']].to_numpy()
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    scaler2 = MinMaxScaler(feature_range=(-100, 100))
    scaler2.fit(X)
    X = scaler2.transform(X)

    df.replace({'name': {'orange': -1, 'grapefruit': 1}}, inplace=True)
    y = df['name'].to_numpy()

    # first we shuffle and build the similarity matrix in order to separate later without shuffling and keep the form of the W matrix
    X, y = shuffle(X, y, random_state=1)
    W = rbf_kernel(X, gamma=1/100)
    X_l, X_u, y_l, y_u , W_l, W_u = train_test_split(X, y, W, test_size = testsize, shuffle=False)

    # delete the first 300 columns from the similarity matrix as we don't need them 
    n = len(y_l)
    W_l = W_l[:,n:]
    W_u = W_u[:,n:]
    return df, W_l, W_u, X_l, X_u, y_l, y_u,max_iter,eps,randomseed

def prepare_data(path):
    config = read_config(path)
    noofsamples = config.get('Noofsamples')
    noofcenters = config.get('Noofcenters')
    nooffeatures = config.get('Nooffeatures')
    testsize = config.get('testsize')
    lr = config.get('learningrate')
    max_iter = config.get('maxiterations')
    eps = config.get('epsilon')
    randomseed = config.get('randomseed')


    X, y = make_blobs(n_samples=noofsamples, centers=noofcenters, n_features=nooffeatures, random_state=123)

    y[y == 0] = -1

    # first we shuffle and build the similarity matrix in order to separate later without shuffling and keep the form of the W matrix
    X, y = shuffle(X, y, random_state=1)
    # W = 1 / (pairwise_distances(X) + 0.0001)
    W  = np.exp(-(euclidean_distances(X, squared=True)) * 0.5)
    X_l, X_u, y_l, y_u , W_l, W_u = train_test_split(X, y, W, test_size = testsize, shuffle=False)

    # delete the first 300 columns from the similarity matrix as we don't need them 
    n = len(y_l)
    W_l = W_l[:,n:]
    W_u = W_u[:,n:]

    return X,y,X_l,X_u,y_l,y_u,W_l,W_u,lr,max_iter,eps,randomseed

