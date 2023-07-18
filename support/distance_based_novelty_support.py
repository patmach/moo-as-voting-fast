import numpy as np

# users_viewed_item is np.array with length == #of items where each entry corresponds
# to the number of users who seen the particular item

def popularity_based_log_novelty_support(user_rating_matrix, items, distance_matrix, k):
    users_list = np.array([[i for i in range(len(user_rating_matrix)) if user_rating_matrix[i]>0.5]])
    if k == 1:   
        debug = np.repeat(np.expand_dims(0, axis=0), users_list.shape[0], axis=0) #/ (distance_matrix.shape[0] - 1)
        return np.repeat(np.expand_dims(0, axis=0), users_list.shape[0], axis=0) #/ (distance_matrix.shape[0] - 1)
    debug=distance_matrix[users_list[:,:k-1,np.newaxis], items].min(axis=1)
    return distance_matrix[users_list[:,:k-1,np.newaxis], items].min(axis=1)