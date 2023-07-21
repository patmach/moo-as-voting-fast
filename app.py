import copy
import main
import numpy as np
import time
import json 
from flask import Flask
from flask import request
import threading

    
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config["DEBUG"] = True
extended_rating_matrix,users_profiles, users_viewed_item, distance_matrix, items,itemIDs, users, userIDs,\
      algorithm_factory, normalizations, args = None, None, None, None, None,None,None, None, None, None, None
computing = False
@app.before_first_request
def init():
    global computing, extended_rating_matrix, users_profiles, users_viewed_item, distance_matrix, items,itemIDs, users, userIDs, algorithm_factory, normalizations, args
    start_time =time.perf_counter()
    if(not computing):
        computing = True
        extended_rating_matrix,users_profiles, users_viewed_item, distance_matrix, items,itemIDs, users, userIDs, \
            algorithm_factory, normalizations, args = main.init()
        computing=False
        print(f"Init done, took: {time.perf_counter() - start_time}")
    threading.Timer(6000.0, init).start()



@app.route('/getRecommendations/<user_id>', methods=["POST", "GET"])
def index(user_id):
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
        if(len(metric_variants)==3):
            if (metric_variants[1] is not None) and (metric_variants[1]!=""):
                client_args.diversity = metric_variants[1]
            if (metric_variants[2] is not None) and (metric_variants[2]!=""):
                client_args.novelty = metric_variants[2]
        userindex = userIDs.index(int(user_id))
        user_mask =  [np.ones(len(items),dtype=np.bool8)]
        if(len(whiteListItemIDs)>0):
            user_mask[0] = np.zeros(len( user_mask[0]))
            for i in whitelistindices:
                user_mask[0][i]=1
        for i in blacklistindices:
            user_mask[0][i]=0
    
        result, support = main.predict_for_user(users[userindex], userindex, items, extended_rating_matrix, users_profiles,\
                        distance_matrix, users_viewed_item, normalizations, np.array(user_mask), algorithm_factory,client_args,\
                              metrics, len(users))
        result = np.squeeze(result)
        for i in range(len(support)):
            for j in range(len(support[0])):
                support[i][j] = max(support[i][j],0) 
        #sums = [sum(support[i]) for i in range(len(result))]
        #result = {itemIDs[int(result[i])]: (support[i]/sums[i]*100).tolist() for i in range(len(result))}
        result = {itemIDs[int(result[i])]: support[i] for i in range(len(result))}
    return json.dumps(result)


@app.route('/x')
def start2():
    main.init()
    return main.init()

app.run(debug=True)

