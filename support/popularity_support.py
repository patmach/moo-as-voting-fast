import numpy as np
avg_ratings = None


def popularity_support(users_viewed_item, compute_for_num_users, num_all_users, popularity_type):
    """

    Parameters
    ----------
    users_viewed_item : np.ndarray
        contains values how many users have rated each item
    compute_for_num_users : int
        number of users the computation should be done for
    num_all_users : int
        number of all users
    popularity_type : str
        code of the popularity variant

    Returns
    -------
    np.ndarray
        computed popularity for all items
    """
    sup =[]
    if (popularity_type == "num_of_ratings"):
        sup = (users_viewed_item / num_all_users)
    elif (popularity_type == "avg_ratings"):
        sup = avg_ratings
    return np.repeat(sup[np.newaxis, :], compute_for_num_users, axis=0) 
   

def set_params_popularity(average_ratings):
    """
    Sets array with average rating of each item

    Parameters
    ----------
    average_ratings : list
        list with average rating of each item
    """
    global avg_ratings
    avg_ratings = np.array(average_ratings)
    