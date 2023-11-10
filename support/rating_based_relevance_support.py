# Returns np.array with shape num_users x num_items which has meaning of support of item for user at step k

def rating_based_relevance_support(rating_matrix):
    """

    Parameters
    ----------
    rating_matrix : np.ndarray
        array with predicted ratings

    Returns
    -------
    np.ndarray
        computed relevance for all items
    """
    return rating_matrix