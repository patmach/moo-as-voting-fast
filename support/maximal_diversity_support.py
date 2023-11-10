import numpy as np

def maximal_diversity_support(users_partial_lists, items, distance_matrix, k):
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
        computed maximal diversity for all items
    """    
    if k == 1:
        return np.repeat(np.expand_dims(distance_matrix.sum(axis=1), axis=0), users_partial_lists.shape[0], axis=0) / (distance_matrix.shape[0] - 1)
    return distance_matrix[users_partial_lists[:,:k-1,np.newaxis], items].min(axis=1)
