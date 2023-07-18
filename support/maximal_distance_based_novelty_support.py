import numpy as np
def maximal_distance_based_novelty_support(user_list, items, distance_matrix, k):
    user_list = np.array(user_list[0]).reshape(1, len(user_list[0]))
    if k == 1:   
        return np.repeat(np.expand_dims(0, axis=0), user_list.shape[0], axis=0) #/ (distance_matrix.shape[0] - 1)
    return distance_matrix[user_list[:,:k-1,np.newaxis], items].min(axis=1)
