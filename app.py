import copy
import sys
import main
import numpy as np
import time
import json 
from flask import Flask
from flask import request
import threading
import datetime

from support.binomial_diversity_support import recompute_user_genres_prob, get_user_genre_prob
import support.calibration_support as calibration_support

    
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config["DEBUG"] = False
extended_rating_matrix, neg_extended_rating_matrix, pos_users_profiles, neg_users_profiles, users_viewed_item,\
      distance_matrix, items,itemIDs, users, userIDs, algorithm_factory, normalizations, args, ease_B, neg_ease_B \
            = None, None, None, None, None, None, None,None,None, None, None, None, None, None, None
lock = threading.Lock()


@app.before_first_request
def init():
    """
    Initialization of the recommender - computation of all needed datasets
    Called before first requets on worker
    """
    global extended_rating_matrix, neg_extended_rating_matrix, pos_users_profiles, neg_users_profiles,\
          users_viewed_item, distance_matrix, items,itemIDs, users, userIDs, algorithm_factory,\
              normalizations, args, ease_B, neg_ease_B
    start_time =time.perf_counter()
    extended_rating_matrix, neg_extended_rating_matrix, pos_users_profiles, neg_users_profiles,\
        users_viewed_item, distance_matrix, items,itemIDs, users, userIDs, algorithm_factory,\
            normalizations, args, ease_B, neg_ease_B = main.init()
    print(f"Init done, took: {time.perf_counter() - start_time}")
    x=datetime.datetime.now()
    y = x + datetime.timedelta(days=1)
    y = y.replace(hour=3, minute=30, second=0, microsecond=0)
    threading.Timer((y - x).seconds, init).start()



@app.route('/getRecommendations/<user_id>', methods=["POST", "GET"])
def index(user_id):
    """
    Process request on recommendations from user

    Parameters
    ----------
    user_id : str
        ID of user that send the request

    Returns
    -------
    dict
        list of recommended items and their supports for each metric
    """
    global extended_rating_matrix, neg_extended_rating_matrix, pos_users_profiles, neg_users_profiles,\
          users_viewed_item, distance_matrix, items,itemIDs, users, userIDs, algorithm_factory,\
            normalizations, args, ease_B, neg_ease_B
    result=[]
    if(request.is_json):
        json_data = request.get_json()
        metrics, whitelistindices, blacklistindices, currentlistindices, metric_variants = get_arguments(json_data)
        client_args = set_client_args(args, json_data, metric_variants)
        user_id = int(user_id)  
        if (user_id not in userIDs):
            create_new_user(user_id)
        userindex = userIDs.index(user_id)      
        rated, scores, pos_rated = get_rated(client_args, user_id, userindex)
        set_rating_matrices_row(rated, scores, client_args, userindex)
        recompute_user_genres_prob(userindex, pos_rated) 
        calibration_support.set_user_genre_prob(userindex, get_user_genre_prob()[userindex])
        user_mask = get_mask(whitelistindices, blacklistindices, currentlistindices)
        result, support = main.predict_for_user(users[userindex], userindex, items, extended_rating_matrix,\
                                            neg_extended_rating_matrix, pos_users_profiles, neg_users_profiles,\
                                                    distance_matrix,\
                                            users_viewed_item, normalizations, np.array(user_mask),\
                                            algorithm_factory,client_args, metrics, len(users),\
                                            currentlistindices)
        result = np.squeeze(result)
        for i in range(len(support)):
            for j in range(len(support[0])):
                support[i][j] = max(np.nan_to_num(support[i][j]),0) 
        result = {itemIDs[int(result[i])]: support[i] for i in range(len(result))}
    return json.dumps(result)


def set_client_args(args, json_data, metric_variants):
    """

    Parameters
    ----------
    client_args : SimpleNamespace
        All arguments of RS
    json_data : dict
        Data in json sent as part of request
    metric_variants : list
        list of codes of selected metrics variants

    Returns
    -------
    SimpleNamespace
        All arguments of RS for this request
    """
    client_args=copy.deepcopy(args)
    client_args.k = json_data["count"]
    client_args.discount_sequences = np.stack([np.geomspace(start=1.0,stop=d**(client_args.k *10),
                                                            num=client_args.k * 10, endpoint=False) 
                                                            for d in client_args.discounts], axis=0)
    if(len(metric_variants)>3):
            if (metric_variants[0] is not None) and (metric_variants[0]!=""):
                client_args.relevance = metric_variants[0]
            if (metric_variants[1] is not None) and (metric_variants[1]!=""):
                client_args.diversity = metric_variants[1]
            if (metric_variants[2] is not None) and (metric_variants[2]!=""):
                client_args.novelty = metric_variants[2]
            if (metric_variants[3] is not None) and (metric_variants[3]!=""):
                client_args.popularity = metric_variants[3]
    return client_args


def get_arguments(json_data): 
    """

    Parameters
    ----------
    json_data : dict
        Data in json sent as part of request

    Returns
    -------
    list
        list of parameters for getting recommendations retrieved from the body of the request
    """
    whiteListItemIDs = json_data["whiteListItemIDs"]
    blackListItemIDs = json_data["blackListItemIDs"]
    curentListItemIDs = json_data["currentListItemIDs"]  
    metrics  = np.array([importance / sum(json_data["metrics"]) for importance in json_data["metrics"]])        
    whitelistindices = [itemIDs.index(int(item_id)) for item_id in whiteListItemIDs if int(item_id) in itemIDs]
    blacklistindices = [itemIDs.index(int(item_id)) for item_id in blackListItemIDs if int(item_id) in itemIDs]
    currentlistindices = [itemIDs.index(int(item_id)) for item_id in curentListItemIDs if int(item_id) in itemIDs]
    metric_variants = json_data["metricVariantsCodes"]
    return metrics, whitelistindices, blacklistindices, currentlistindices, metric_variants


def get_rated(client_args, user_id, userindex):
    """

    Parameters
    ----------
    client_args : : SimpleNamespace
        All arguments of RS
    user_id : int
        ID of user that send the request
    userindex : int
        index of the user in datasets

    Returns
    -------
    list
        list of items rated by user, list of rating values, list of items positively rated by user
    """
    global pos_users_profiles, neg_users_profiles
    neg_rated = main.get_ratings_of_user(args.connectionstring, user_id,\
                                        only_positive=False)
    pos_rated = neg_rated[neg_rated["ratingscore"] > 5]
    pos_scores = list(pos_rated["ratingscore"].apply(lambda x: x/10.0 if x>5 else 0))
    neg_scores = list(neg_rated["ratingscore"].apply(lambda x: (x - 5) * 2 /10.0))
    pos_rated = list(pos_rated["itemid"])
    pos_rated = list(map(itemIDs.index, pos_rated))
    neg_rated = list(neg_rated["itemid"])
    neg_rated = list(map(itemIDs.index, neg_rated))
    rated = pos_rated
    scores = pos_scores
    if (client_args.relevance == "also_negative"):
        rated = neg_rated
        scores = neg_scores
    pos_users_profiles[userindex] = pos_rated
    neg_users_profiles[userindex] = neg_rated
    return rated, scores, pos_rated


def create_new_user(user_id):
    """
    Creates new user and add him to every dataset

    Parameters
    ----------
    user_id : int
        ID of user that send the request
    """
    global extended_rating_matrix, neg_extended_rating_matrix, userIDs, users, pos_users_profiles, neg_users_profiles
    lock.acquire()
    try:
        extended_rating_matrix = np.append(extended_rating_matrix, [np.zeros(len(items),dtype=np.float32)], axis=0)
        neg_extended_rating_matrix = np.append(neg_extended_rating_matrix, [np.zeros(len(items),dtype=np.float32)], axis=0)
        userIDs.append(user_id)
        users = np.append(users,[len(users)])
        get_user_genre_prob().append(np.array([0]* len(get_user_genre_prob()[0]))) 
        calibration_support.get_user_genre_prob().append(np.array([0]* len(get_user_genre_prob()[0]))) 
        pos_users_profiles = np.append(pos_users_profiles,[-1])
        neg_users_profiles = np.append(neg_users_profiles,[-1])
    except:
        debug =1
    finally:
        lock.release()


def set_rating_matrices_row(rated, scores, client_args, userindex):
    """
    Sets new predtiction values in matrix row corresponding to user

    Parameters
    ----------
    rated : list
        list of items rated by user
    scores : list
        list of rating values
    client_args : SimpleNamespace
        All arguments of RS
    userindex : int
        index of the user in datasets
    """
    global extended_rating_matrix, neg_extended_rating_matrix
    user_X = np.zeros(len(items),dtype=np.float64)
    user_X[rated] = scores   
    if(client_args.relevance == "also_negative"):
        neg_extended_rating_matrix[userindex] = user_X.dot(neg_ease_B)
    else:
        extended_rating_matrix[userindex] = user_X.dot(ease_B)


def get_mask(whitelistindices, blacklistindices, currentlistindices):
    """

    Parameters
    ----------
    whitelistindices : list
        list of items that can be recommended to user
    blacklistindices : list
        list of items that can't be recommended to user
    currentlistindices : list
        list of items that are already part of the list of the recommendations

    Returns
    -------
    np.ndarray
        array that contains 1 in the indices of items that can be recommended, 0 otherwise
    """
    user_mask = [np.ones(len(items),dtype=np.bool8)]
    if(len(whitelistindices)>0):
        user_mask = [np.zeros(len( user_mask[0]),dtype=np.bool8)]
        for i in whitelistindices:
            user_mask[0][i]=True
    for i in blacklistindices:
        user_mask[0][i]=False
    for i in currentlistindices:
        user_mask[0][i]=False
    return user_mask


@app.route('/start')
def start():
    return ""


if __name__ == "__main__":
    app.run(threaded=True)

