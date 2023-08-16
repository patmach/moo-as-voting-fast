import copy
import main
import numpy as np
import time
import json 
from flask import Flask
from flask import request
import threading

from support.binomial_diversity_support import recompute_user_genres_prob, get_user_genre_prob

    
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config["DEBUG"] = True
extended_rating_matrix,users_profiles, users_viewed_item, distance_matrix, items,itemIDs, users, userIDs,\
      algorithm_factory, normalizations, args, ease_B = None, None, None, None, None,None,None, None, None, None, None, None
computing = False

lock = threading.Lock()

@app.before_first_request
def init():
    global computing, extended_rating_matrix, users_profiles, users_viewed_item, distance_matrix, items,itemIDs, users, userIDs,\
          algorithm_factory, normalizations, args, ease_B
    start_time =time.perf_counter()
    if(not computing):
        computing = True
        extended_rating_matrix,users_profiles, users_viewed_item, distance_matrix, items,itemIDs, users, userIDs, \
            algorithm_factory, normalizations, args, ease_B = main.init()
        computing=False
        print(f"Init done, took: {time.perf_counter() - start_time}")
    threading.Timer(3600.0, init).start()



@app.route('/getRecommendations/<user_id>', methods=["POST", "GET"])
def index(user_id):
    global extended_rating_matrix, users_profiles, users_viewed_item, distance_matrix, items,itemIDs, users, userIDs,\
          algorithm_factory, normalizations, args, ease_B
    json_data = []
    result=[]
    client_args=copy.deepcopy(args)
    if(request.is_json):
        json_data = request.get_json()
        whiteListItemIDs = json_data["whiteListItemIDs"]
        blackListItemIDs = json_data["blackListItemIDs"]
        client_args.k = json_data["count"]
        client_args.discount_sequences = np.stack([np.geomspace(start=1.0,stop=d**client_args.k, num=client_args.k, endpoint=False) for d in client_args.discounts], axis=0)
        metrics  = np.array([importance / sum(json_data["metrics"]) for importance in json_data["metrics"]])
        
        whitelistindices = [itemIDs.index(int(item_id)) for item_id in whiteListItemIDs if int(item_id) in itemIDs]
        blacklistindices = [itemIDs.index(int(item_id)) for item_id in blackListItemIDs if int(item_id) in itemIDs]
        metric_variants = json_data["metricVariantsCodes"]
        if(len(metric_variants)>2):
            if (metric_variants[1] is not None) and (metric_variants[1]!=""):
                client_args.diversity = metric_variants[1]
            if (metric_variants[2] is not None) and (metric_variants[2]!=""):
                client_args.novelty = metric_variants[2]
        user_id = int(user_id)
        rated = main.get_positive_ratings_of_user(args.connectionstring, user_id)
        scores = list(rated["ratingscore"].apply(lambda x: x/10 if x>5 else 0))
        rated = list(rated["itemid"])
        rated = list(map(itemIDs.index, rated))
        user_X = np.zeros(len(items),dtype=np.int32)
        user_X[rated] = scores   
        if (user_id not in userIDs):
            lock.acquire()
            try:
                extended_rating_matrix = np.append(extended_rating_matrix, [np.zeros(len(items),dtype=np.float32)], axis=0)
                userIDs.append(user_id)
                users = np.append(users,[len(users)])
                get_user_genre_prob().append(np.array([0]* len(get_user_genre_prob()[0]))) 
                users_profiles = np.append(users_profiles,[-1])
            except:
                debug =1
            finally:
                lock.release()
        userindex = userIDs.index(user_id)         
        users_profiles[userindex] = rated
        extended_rating_matrix[userindex] = user_X.dot(ease_B)
        if (client_args.diversity == "binomial_diversity"):
            recompute_user_genres_prob(userindex, rated)
        user_mask = [np.ones(len(items),dtype=np.bool8)]
        if(len(whiteListItemIDs)>0):
            user_mask = [np.zeros(len( user_mask[0]),dtype=np.bool8)]
            for i in whitelistindices:
                user_mask[0][i]=True
        for i in blacklistindices:
            user_mask[0][i]=False
    
        result, support = main.predict_for_user(users[userindex], userindex, items, extended_rating_matrix, users_profiles,\
                        distance_matrix, users_viewed_item, normalizations, np.array(user_mask), algorithm_factory,client_args,\
                              metrics, len(users))
        result = np.squeeze(result)
        for i in range(len(support)):
            for j in range(len(support[0])):
                support[i][j] = max(support[i][j],0) 
        result = {itemIDs[int(result[i])]: support[i] for i in range(len(result))}
    return json.dumps(result)


@app.route('/x')
def start2():
    main.init()
    return main.init()

app.run(debug=True)

