import numpy as np
# old_obj_values are diversityy values for lists of length k - 1
def intra_list_distance_based_novelty_support(user_list, items, distance_matrix):
    user_list = np.array(user_list[0]).reshape(1, len(user_list[0]))
    return distance_matrix[user_list[:,:,np.newaxis], items].sum(axis=1) / user_list.shape[1]
