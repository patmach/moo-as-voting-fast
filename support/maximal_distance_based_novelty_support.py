import numpy as np
def maximal_distance_based_novelty_support(user_list, items, distance_matrix, k):
    user_list = np.array(user_list[0]).reshape(1, len(user_list[0]))
    debug = distance_matrix[user_list[:,:,np.newaxis], items].min(axis=1)
    return distance_matrix[user_list[:,:,np.newaxis], items].min(axis=1)
