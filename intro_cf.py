# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('ml-100k/u.data', sep='\t', names=names)

n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print str(n_users) + 'users'
print str(n_items) + 'items'

ratings = np.zeros((n_users, n_items))
for row in df.itertuples():
    ratings[row[1]-1, row[2]-1] = row[3]
    
sparsity = float(len(ratings.nonzero()[0]))
sparsity /= (ratings.shape[0] * ratings.shape[1])
sparsity *= 100
print(sparsity)

def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in xrange(ratings.shape[0]):
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0],
                                        size=10, replace = False)
        train[user, test_ratings] = 0
        test[user, test_ratings] = ratings[user, test_ratings]
    
    assert(np.all(train * test) == 0)
    return train, test

train, test = train_test_split(ratings)
        
def slow_similarity(ratings, kind='user'):
    if kind == 'user':
        axmax = 0
        axmin = 1
    elif kind == 'item':
        axmax = 1
        axmin = 0
    sim = np.zeros((ratings.shape[axmax], ratings.shape[axmax]))
    for u in xrange(ratings.shape[axmax]):
        for uprime in xrange(ratings.shape[axmax]):
            rui_sqrd = 0.
            ruprimei_sqrd = 0.
            for i in xrange(ratings.shape[axmax]):
                sim[u, uprime] = ratings[u, i] * ratings[uprime, i]
                rui_sqrd += ratings[u, i] ** 2
                ruprimei_sqrd += ratings[uprime, i] ** 2
            sim[u, uprime] /= rui_sqrd * ruprimei_sqrd
    return sim

def fast_similarity(ratings, kind='user', epsilon=0.9):
    # epsilon -> small number for handling dived-by-zero errors
    if kind == 'user':
        sim = ratings.dot(ratings.T) + epsilon
    elif kind == 'item':
        sim = ratings.T.dot(ratings) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim/norms/norms.T)

user_similarity = fast_similarity(train, kind='user')
item_similarity = fast_similarity(train, kind='item')
#print(user_similarity[:4, :4])
#print(item_similarity[:4, :4])

def predict_slow_simple(ratings, similarity, kind='user'):
    pred = np.zeros(ratings.shape)
    if kind == 'user':
        for i in xrange(ratings.shape[0]):
            for j in xrange(ratings.shape[1]):
                pred[i, j] = similarity[i, :].dot(ratings[:, j])\
                             /np.sum(np.abs(similarity[i, :]))
        return pred
    elif kind == 'item':
        for i in xrange(ratings.shape[0]):
            for j in xrange(ratings.shape[1]):
                pred[i, j] = similarity[j, :].dot(ratings[i, :].T)\
                             /np.sum(np.abs(similarity[j, :]))

        return pred
        
def predict_fast_simple(ratings, similarity, kind='user'):
    if kind == 'user':
        return similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif kind == 'item':
        return ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
        
from sklearn.metrics import mean_squared_error
def get_mse(pred, actual):
    #ignore nonzero terms
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

def get_rmse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual) ** 0.5

item_prediction = predict_fast_simple(ratings, item_similarity, kind='item')
user_prediction = predict_fast_simple(ratings, user_similarity, kind='user')

print 'User-based CF MSE in train: ' + str(get_mse(user_prediction, train))
print 'User-based CF rMSE in train: ' + str(get_rmse(user_prediction, train))
print 'Item-based CF MSE in train: ' + str(get_mse(item_prediction, train))
print 'Item-based CF rMSE in train: ' + str(get_rmse(item_prediction, train))
print 
print 'User-based CF MSE in test: ' + str(get_mse(user_prediction, test))
print 'User-based CF RMSE in test: ' + str(get_rmse(user_prediction, test))
print 'Item-based CF MSE in test: ' + str(get_mse(item_prediction, test))
print 'Item-based CF RMSE in test: ' + str(get_rmse(item_prediction, test))
print 'cf mse: '+ str((get_rmse(user_prediction, test)+get_rmse(item_prediction, test))/2)
"""      
def predict_topk(ratings, similarity, kind='user', k=40):
    pred = np.zeros(ratings.shape)
    if kind == 'user':
        for i in xrange(ratings.shape[0]):
            top_k_users = [np.argsort(similarity[:,i])[:-k-1:-1]]
            for j in xrange(ratings.shape[1]):
                pred[i, j] = similarity[i, :][top_k_users].dot(ratings[:, j][top_k_users]) 
                pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))
    if kind == 'item':
        for j in xrange(ratings.shape[1]):
            top_k_items = [np.argsort(similarity[:,j])[:-k-1:-1]]
            for i in xrange(ratings.shape[0]):
                pred[i, j] = similarity[j, :][top_k_items].dot(ratings[i, :][top_k_items].T) 
                pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items]))        
    
    return pred        

def predict_fast_topk(ratings, similarity, kind='user', k=40):
    
    return

#pred = predict_topk(train, user_similarity, kind='user', k=50)
#print 'Top-k User-based CF MSE: ' + str(get_mse(pred, test))

#pred = predict_topk(train, item_similarity, kind='item', k=15)
#print 'Top-k Item-based CF MSE: ' + str(get_mse(pred, test))
    
def predict_nobias(ratings, similarity, kind='user'):
    if kind == 'user':
        user_bias = ratings.mean(axis=1)
        ratings = (ratings - user_bias[:, np.newaxis]).copy()
        pred = similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
        pred += user_bias[:, np.newaxis]
    elif kind == 'item':
        item_bias = ratings.mean(axis=0)
        ratings = (ratings - item_bias[np.newaxis, :]).copy()
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
        pred += item_bias[np.newaxis, :]
        
    return pred

#user_pred = predict_nobias(train, user_similarity, kind='user')
#print 'Bias-subtracted User-based CF MSE: ' + str(get_mse(user_pred, test))

#item_pred = predict_nobias(train, item_similarity, kind='item')
#print 'Bias-subtracted Item-based CF MSE: ' + str(get_mse(item_pred, test))

def predict_topk_nobias(ratings, similarity, kind='user', k=40):
    pred = np.zeros(ratings.shape)
    if kind == 'user':
        user_bias = ratings.mean(axis=1)
        ratings = (ratings - user_bias[:, np.newaxis]).copy()
        for i in xrange(ratings.shape[0]):
            top_k_users = [np.argsort(similarity[:,i])[:-k-1:-1]]
            for j in xrange(ratings.shape[1]):
                pred[i, j] = similarity[i, :][top_k_users].dot(ratings[:, j][top_k_users]) 
                pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))
        pred += user_bias[:, np.newaxis]
    if kind == 'item':
        item_bias = ratings.mean(axis=0)
        ratings = (ratings - item_bias[np.newaxis, :]).copy()
        for j in xrange(ratings.shape[1]):
            top_k_items = [np.argsort(similarity[:,j])[:-k-1:-1]]
            for i in xrange(ratings.shape[0]):
                pred[i, j] = similarity[j, :][top_k_items].dot(ratings[i, :][top_k_items].T) 
                pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items])) 
        pred += item_bias[np.newaxis, :]
        
    return pred   
 
k_array = [5, 15, 30, 50, 100, 200]
user_train_mse = []
user_test_mse = []
item_test_mse = []
item_train_mse = []

for k in k_array:
    user_pred = predict_topk_nobias(train, user_similarity, kind='user', k=k)
    item_pred = predict_topk_nobias(train, item_similarity, kind='item', k=k)
    
    user_train_mse += [get_mse(user_pred, train)]
    user_test_mse += [get_mse(user_pred, test)]
    
    item_train_mse += [get_mse(item_pred, train)]
    item_test_mse += [get_mse(item_pred, test)]  
                      
    
idx_to_movie = {}
with open('ml-100k/u.item', 'r') as f:
    for line in f.readlines():
        info = line.split('|')
        idx_to_movie[int(info[0])-1] = info[4]
        
def top_k_movies(similarity, mapper, movie_idx, k=6):
    return [mapper[x] for x in np.argsort(similarity[movie_idx,:])[:-k-1:-1]]   

idx = 0 # Toy Story
movies = top_k_movies(item_similarity, idx_to_movie, idx)
for movie in movies:
    print movie
print '\n'



from sklearn.metrics import pairwise_distances
# Convert from distance to similarity
item_correlation = 1 - pairwise_distances(train.T, metric='correlation')
item_correlation[np.isnan(item_correlation)] = 0.
idx = 0 # Toy Story
movies = top_k_movies(item_correlation, idx_to_movie, idx)
for movie in movies:
    print movie    
print '\n'                 
    
idx = 1 # GoldenEye
movies = top_k_movies(item_similarity, idx_to_movie, idx)    
for movie in movies:
    print movie    
print '\n'

idx = 1 # GoldenEye
movies = top_k_movies(item_correlation, idx_to_movie, idx)
for movie in movies:
    print movie 
print '\n'

idx = 20 # Muppet Treasure Island
movies = top_k_movies(item_similarity, idx_to_movie, idx)
for movie in movies:
    print movie    
print '\n'

idx = 20 # Muppet Treasure Island
movies = top_k_movies(item_correlation, idx_to_movie, idx)
for movie in movies:
    print movie    
print '\n'
"""    
    
        
        
        