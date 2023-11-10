import numpy as np

def popularity_complement_support(users_viewed_item, compute_for_num_users, num_all_users):
    """

    Parameters
    ----------
    users_viewed_item : np.ndarray
        contains values how many users have rated each item
    compute_for_num_users : int
        number of users the computation should be done for
    num_all_users : int
        number of all users

    Returns
    -------
    np.ndarray
        computed popularity complement novelty for all items
    """
    sup = 1.0 - (users_viewed_item / num_all_users)
    return np.repeat(sup[np.newaxis, :], compute_for_num_users, axis=0) # proper expand_dims to ensure later broadcasting
   
