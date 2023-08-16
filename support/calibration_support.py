import numpy as np

user_genre_prob=None
metadata_matrix=None

def set_params_calibration(user_genre_prob_param,metadata_matrix_param,):
    global user_genre_prob, metadata_matrix
    user_genre_prob = user_genre_prob_param
    metadata_matrix = metadata_matrix_param

def calibration_support_for_item(users_partial_lists, item, p_prob, q_prob, k, alpha = 0.01):
    users_partial_lists[:,k-1] = item
    used_genres = metadata_matrix[users_partial_lists[:,:k,np.newaxis], :].sum(axis=1)
    used_genres_sum = used_genres.sum()
    if (used_genres_sum > 0):
        q_prob += (1-alpha) * (used_genres.flatten() / used_genres_sum)
    result = p_prob * np.log2(np.divide(p_prob, q_prob, where=q_prob!=0), where=p_prob>0)
    calibration = np.abs(result.sum())
    return - calibration 

def calibration_support(users_partial_lists, items, user_index, k, alpha = 0.01):
    p_prob = user_genre_prob[user_index]
    sum_user_genre_prob = user_genre_prob[user_index].sum()
    if sum_user_genre_prob > 0:
        p_prob = user_genre_prob[user_index] / sum_user_genre_prob
    else: 
        p_prob = np.zeros(len(user_genre_prob[0]), dtype=np.float32)
    q_prob = alpha * p_prob
    result = np.array([calibration_support_for_item(users_partial_lists, item, p_prob,q_prob,k, alpha) for item in items])
    if (k == 1):
        return np.expand_dims(result - result.min(),axis=0)
    current = calibration_support_for_item(users_partial_lists, users_partial_lists[:,k-2], p_prob,q_prob, k-1, alpha)
    return np.expand_dims(result - current,axis=0)
    