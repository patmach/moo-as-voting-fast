import numpy as np

# users_viewed_item is np.array with length == #of items where each entry corresponds
# to the number of users who seen the particular item

def popularity_based_log_novelty_support(users_viewed_item, compute_for_num_users, num_all_users):
    sup = - np.log2(users_viewed_item / num_all_users)
    return np.repeat(sup[np.newaxis, :], compute_for_num_users, axis=0) # proper expand_dims to ensure later broadcasting
   
