import numpy as np
def maximal_diversity_support(users_partial_lists, items, distance_matrix, k):
    if k == 1:   
        debug = np.repeat(np.expand_dims(distance_matrix.sum(axis=1), axis=0), users_partial_lists.shape[0], axis=0) / (distance_matrix.shape[0] - 1)
        return np.repeat(np.expand_dims(distance_matrix.sum(axis=1), axis=0), users_partial_lists.shape[0], axis=0) / (distance_matrix.shape[0] - 1)
    debug=distance_matrix[users_partial_lists[:,:k-1,np.newaxis], items].min(axis=1)
    return distance_matrix[users_partial_lists[:,:k-1,np.newaxis], items].min(axis=1)
