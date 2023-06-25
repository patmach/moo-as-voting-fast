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
extended_rating_matrix, users_viewed_item, distance_matrix, items,itemIDs, users, userIDs,\
      mask, algorithm_factory, normalizations, args = None, None, None, None, None,None,None, None, None, None, None
computing = False
@app.before_first_request
def init():
    global computing, extended_rating_matrix, users_viewed_item, distance_matrix, items,itemIDs, users, userIDs, mask, algorithm_factory, normalizations, args
    start_time =time.perf_counter()
    if(not computing):
        computing = True
        extended_rating_matrix, users_viewed_item, distance_matrix, items,itemIDs, users, userIDs, mask, algorithm_factory, normalizations, args = main.init()
        computing=False
        print(f"Init done, took: {time.perf_counter() - start_time}")
    threading.Timer(60.0, init).start()



@app.route('/train')
def start():
    global extended_rating_matrix, users_viewed_item, distance_matrix, items,itemIDs, users, user_IDS, mask, algorithm_factory, normalizations, args
    extended_rating_matrix, users_viewed_item, distance_matrix, items, itemIDs, users, user_IDS, mask, algorithm_factory, normalizations, args = main.init()


@app.route('/getRecommendations/<user_id>', methods=["POST", "GET"])
def index(user_id):
    json_data = []
    result=[]
    if(request.is_json):
        json_data = request.get_json()
        whiteListItemIDs = json_data["whiteListItemIDs"]
        blackListItemIDs = json_data["blackListItemIDs"]
        args.k = json_data["count"]
        args.discount_sequences = np.stack([np.geomspace(start=1.0,stop=d**args.k, num=args.k, endpoint=False) for d in args.discounts], axis=0)
        metrics  = np.array([importance / sum(json_data["metrics"]) for importance in json_data["metrics"]])
        
        whitelistindices = [itemIDs.index(int(item_id)) for item_id in whiteListItemIDs if int(item_id) in itemIDs]
        blacklistindices = [itemIDs.index(int(item_id)) for item_id in blackListItemIDs if int(item_id) in itemIDs]

        userindex = userIDs.index(int(user_id))
        user_mask =  mask[userindex:userindex+1].copy()
        if(len(whiteListItemIDs)>0):
            user_mask[0] = np.zeros(len( user_mask[0]))
            for i in whitelistindices:
                user_mask[0][i]=1
        for i in blacklistindices:
            user_mask[0][i]=0
    
        result, support = main.predict_for_user(users[userindex], userindex, items, extended_rating_matrix, distance_matrix,\
                        users_viewed_item, normalizations, user_mask, algorithm_factory,args, metrics, len(users))
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

