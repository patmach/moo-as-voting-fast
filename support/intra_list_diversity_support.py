import numpy as np
# old_obj_values are diversityy values for lists of length k - 1
def intra_list_diversity_support(users_partial_lists, items, distance_matrix, k):
    """

    Parameters
    ----------
    users_partial_lists : np.ndarray
        partial list of recommendations
    items : np.ndarray
        array of all items
    distance_matrix : np.ndarray
        matrix item x item with values representing distance (size of difference) of the 2 items
    k : int
        rank of the item in list

    Returns
    -------
    np.ndarray
        computed intra list diversity for all items
    """    
    if k == 1:
        return np.repeat(np.expand_dims(distance_matrix.sum(axis=1), axis=0), users_partial_lists.shape[0], axis=0) / (distance_matrix.shape[0] - 1)
    elif k == 2:
        return 2 * distance_matrix[users_partial_lists[:,:k-1,np.newaxis], items].sum(axis=1) / k
    return distance_matrix[users_partial_lists[:,:k-1,np.newaxis], items].sum(axis=1) / (k - 1)
