import numpy as np
# old_obj_values are diversityy values for lists of length k - 1
def intra_list_distance_based_novelty_support(user_list, items, distance_matrix, k):
    user_list = np.array(user_list[0]).reshape(1, len(user_list[0]))
    if k == 1:
        debug= np.repeat(np.expand_dims(distance_matrix.sum(axis=1), axis=0), user_list.shape[0], axis=0) / (distance_matrix.shape[0] - 1)
        return np.repeat(np.expand_dims(distance_matrix.sum(axis=1), axis=0), user_list.shape[0], axis=0) / (distance_matrix.shape[0] - 1)
    elif k == 2:
#        old_supp = distance_matrix[users_partial_lists[:, 0]].sum(axis=1, keepdims=True) / (distance_matrix.shape[0] - 1) # TODO remove, probably not necessary as it is constant per user
        debug=2 * distance_matrix[user_list[:,:k-1,np.newaxis], items].sum(axis=1) / k #- old_supp
        return 2 * distance_matrix[user_list[:,:k-1,np.newaxis], items].sum(axis=1) / k #- old_supp
    debug= distance_matrix[user_list[:,:k-1,np.newaxis], items].sum(axis=1) / (k - 1)
    return distance_matrix[user_list[:,:k-1,np.newaxis], items].sum(axis=1) / (k - 1)
