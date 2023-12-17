import numpy as np

user_genre_prob=None
metadata_matrix=None

def get_user_genre_prob():
    """

    Returns
    -------
    np.ndarray
        array with arrays of users probability of genres
    """
    global user_genre_prob
    return user_genre_prob


def set_user_genre_prob(user_index, value):
    """
    recomputes probability of one genre of one user

    Parameters
    ----------
    user_index : int
        index of this user in arrays
    rated : list
        list of items rated by user
    """
    global user_genre_prob
    user_genre_prob[user_index] = value


def set_params_calibration(user_genre_prob_param,metadata_matrix_param):
    """Sets parameters for binomial diversity

    Parameters
    ----------
    user_genre_prob_param : np.ndarray
        array with arrays of users probability of genres
    metadata_matrix_param : np.ndarray
        array with arrays of items and their affiliation to the genre
    """
    global user_genre_prob, metadata_matrix
    user_genre_prob = user_genre_prob_param
    metadata_matrix = metadata_matrix_param


def calibration_support_for_item(users_partial_lists, item, p_prob, k, alpha = 0.01):
    """
    Computes calibration support for one item in given list of recommendations

    Parameters
    ----------
    users_partial_lists : np.ndarray
        partial list of recommendations
    item : int
        index of item
    p_prob : np.ndarray
        probabilities of genres for the selected user
    k : int
        rank of the item in list
    alpha : float, optional
        sets the value of p_prob in computation of q_prob, by default 0.01

    Returns
    -------
    int
        computed calibration
    """
    users_partial_lists[:,k-1] = item
    used_genres = metadata_matrix[users_partial_lists[:,:k,np.newaxis], :].sum(axis=1)
    used_genres_sum = used_genres.sum()
    q_prob = alpha * p_prob
    if (used_genres_sum > 0):
        q_prob += (1-alpha) * (used_genres.flatten() / k)
    result = p_prob * np.log2(np.divide(p_prob, q_prob, where=q_prob!=0), where=p_prob>0)
    calibration = np.abs(result.sum())
    if np.isnan(calibration):# can't be computed
        return - 1000.0 # small magic number
    else: 
        return - calibration


def calibration_support(users_partial_lists, items, user_index, k, alpha = 0.01):
    """
    Computes contribution to calibration in given list of recommendations for all items


    Parameters
    ----------
    users_partial_lists : np.ndarray
        partial list of recommendations
    items : np.ndarray
        array of all items
    user_index : int
        index of user
    k : int
        rank of the item in list
    alpha : float, optional
        sets the value of p_prob in computation of q_prob, by default 0.01

    Returns
    -------
    np.ndarray
        computed calibration for all items
    """
    p_prob = user_genre_prob[user_index]
    result = np.array([calibration_support_for_item(users_partial_lists, item, p_prob,k, alpha) for item in items])
    if (k == 1):
        return np.expand_dims(result - result.min(),axis=0)
    current = calibration_support_for_item(users_partial_lists, users_partial_lists[:,k-2], p_prob, k-1, alpha)
    return np.expand_dims(result - current,axis=0)
    