import numpy as np

def maximal_distance_based_novelty_support(user_list, items, distance_matrix):
    """

    Parameters
    ----------
    user_list : np.ndarray
        list of all items rated by user
    items : np.ndarray
        array of all items
    distance_matrix : np.ndarray
        matrix item x item with values representing distance (size of difference) of the 2 items

    Returns
    -------
    np.ndarray
        computed maximal distance based novelty for all items
    """
    user_list = np.array(user_list[0]).reshape(1, len(user_list[0]))
    if (user_list.shape[1] == 0):
        return np.array([1.0] * len(items)).reshape(1, len(items))
    return distance_matrix[user_list[:,:,np.newaxis], items].min(axis=1)
