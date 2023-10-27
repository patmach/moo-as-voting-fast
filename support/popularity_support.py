import numpy as np
avg_ratings = None

# users_viewed_item is np.array with length == #of items where each entry corresponds
# to the number of users who seen the particular item
def popularity_support(users_viewed_item, compute_for_num_users, num_all_users, popularity_type):
    sup =[]
    if (popularity_type == "num_of_ratings"):
        sup = (users_viewed_item / num_all_users)
    elif (popularity_type == "avg_ratings"):
        sup = avg_ratings
    debug = np.repeat(sup[np.newaxis, :], compute_for_num_users, axis=0)
    return np.repeat(sup[np.newaxis, :], compute_for_num_users, axis=0) # proper expand_dims to ensure later broadcasting
   
def set_params_popularity(average_ratings):
    global avg_ratings
    avg_ratings = np.array(average_ratings)
    