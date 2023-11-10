import random
import numpy as np
import math 

alpha=0.5
user_genre_prob=None
genres_prob_all=None
metadata_matrix=None
genre_to_genre_id = None
all_genres=None


def get_user_genre_prob():
    """

    Returns
    -------
    np.ndarray
        array with arrays of users probability of genres
    """
    global user_genre_prob
    return user_genre_prob


def set_params_bin_diversity(user_genre_prob_param,genres_prob_all_param,metadata_matrix_param,genre_to_genre_id_param,\
                             all_genres_param):
    """
    Sets parameters for binomial diversity

    Parameters
    ----------
    user_genre_prob_param : np.ndarray
        array with arrays of users probability of genres
    genres_prob_all_param : np.ndarray
        array probability of genres per all users
    metadata_matrix_param : np.ndarray
        array with arrays of items and their affiliation to the genre
    genre_to_genre_id_param : dict
        mapping from genre to its id
    all_genres_param : list
        list of all genres
    """
    global user_genre_prob, genres_prob_all,metadata_matrix,genre_to_genre_id, all_genres
    user_genre_prob = user_genre_prob_param
    genres_prob_all = genres_prob_all_param
    metadata_matrix = metadata_matrix_param
    genre_to_genre_id = genre_to_genre_id_param
    all_genres = all_genres_param


def recompute_user_genres_prob(user_index, rated):
    """
    recomputes probability of one genre of one user

    Parameters
    ----------
    user_index : int
        index of this user in arrays
    rated : list
        list of items rated by user
    """
    user_genre_prob[user_index] = np.zeros(len(genres_prob_all), dtype=float)
    for j in rated:
        user_genre_prob[user_index]+=metadata_matrix[j]
    user_genre_prob[user_index] /= len(rated)


def binomial_diversity_support_for_item(users_partial_lists, item, user_index, k):
    """
    Computes binomial diversity for one item in given list of recommendations

    Parameters
    ----------
    users_partial_lists : np.ndarray
        partial list of recommendations
    item : int
        index of item
    user_index : int
        index of user
    k : int
        rank of the item in list

    Returns
    -------
    int
        computed binomial diversity
    """
    users_partial_lists[:,k-1] = item
    probabilities = (1- alpha) * genres_prob_all + alpha*user_genre_prob[user_index]
    used_genres = metadata_matrix[users_partial_lists[:,:k,np.newaxis], :].sum(axis=1).flatten()
    coverage = np.prod((1 - np.array([probabilities[i] for i in range(len(used_genres)) if used_genres[i] == 0])) ** k) \
                **(1.0/len(genre_to_genre_id))
    
    prob_redundancy = np.array([probabilities[i] for i in range(len(used_genres)) if used_genres[i] > 0])
    values_redundancy = np.array([used_genres[i] for i in range(len(used_genres)) if used_genres[i] > 0])   
    more_than_zero = 1 - ((1-prob_redundancy)**k)
    sums_redundancy = np.array([sum([math.comb(k, j)*(prob_redundancy[i]**j)*((1-prob_redundancy[i])**(k-j)) / more_than_zero[i]\
                                      for j in range(1,values_redundancy[i])]) for i in range(len(values_redundancy))])
    non_redundancy = np.prod(1 - sums_redundancy) ** (1.0/values_redundancy.shape[0]) if values_redundancy.shape[0] > 0 else 1
    return coverage*non_redundancy


def binomial_diversity_support(users_partial_lists, items, user_index, k, n_items_to_compute = 0):   
    """
    Computes contribution to binomial diversity  in given list of recommendations for all items

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
    n_items_to_compute : int, optional
        if > 0 the binomial diversity is computed for only n randomly selected items, by default 0

    Returns
    -------
    np.ndarray
        computed binomial diversity for all items
    """
    items_to_compute = items
    if (n_items_to_compute > 0):
        items_to_compute = random.sample(list(items), k=n_items_to_compute)
    result = np.array([binomial_diversity_support_for_item(users_partial_lists, item, user_index,k) 
                       if (item in items_to_compute) else -1 for item in items])
    if (k == 1):
        return np.expand_dims(result - result.min(),axis=0)
    current = binomial_diversity_support_for_item(users_partial_lists, users_partial_lists[:,k-2], user_index,k-1)
    return np.expand_dims(result - current,axis=0)
    

