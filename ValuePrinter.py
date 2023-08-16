import main
import copy
import numpy as np
import random
import pandas as pd
import sys
import os

def predict_for_one_user(user_id, diversity_type, novelty_type, k, importances):
    result=[]
    client_args=copy.deepcopy(args)
    whiteListItemIDs = []
    client_args.k = k
    client_args.discount_sequences = np.stack([np.geomspace(start=1.0,stop=d**client_args.k, num=client_args.k, endpoint=False) for d in client_args.discounts], axis=0)
    metrics  = np.array([importance / sum(importances) for importance in importances])
    user_id = int(user_id)
    userindex = userIDs.index(user_id)
    blacklistindices = users_profiles[userindex]
    rated = blacklistindices
    blacklistindices=np.append(blacklistindices,np.array(random.choices(items, k=random.randint(0, 50))))
    whitelistindices = [itemIDs.index(int(item_id)) for item_id in whiteListItemIDs if int(item_id) in itemIDs]
    client_args.diversity = diversity_type
    client_args.novelty = novelty_type
    
    user_X = np.zeros(len(items),dtype=np.int32)
    user_X[rated] = 1
    extended_rating_matrix[userindex] = user_X.dot(ease_B)
    user_mask = [np.ones(len(items),dtype=np.bool8)]
    if(len(whiteListItemIDs)>0):
        user_mask = [np.zeros(len( user_mask[0]),dtype=np.bool8)]
        for i in whitelistindices:
            user_mask[0][i]=True
    for i in list(map(int,blacklistindices)):
        user_mask[0][i]=False

    result, support = main.predict_for_user(users[userindex], userindex, items, extended_rating_matrix, users_profiles,\
                    distance_matrix, users_viewed_item, normalizations, np.array(user_mask), algorithm_factory,client_args,\
                            metrics, len(users))
    return support


extended_rating_matrix,users_profiles, users_viewed_item, distance_matrix, items,itemIDs, users, userIDs, \
            algorithm_factory, normalizations, args, ease_B = main.init()
k = 15
values_dfs = []
for diversity_type in ["intra_list_diversity","maximal_diversity","binomial_diversity"]:
    for novelty_type in ["popularity_complement","maximal_distance_based_novelty","intra_list_distance_based_novelty"]:
        ranks=[]
        supports_results=[[],[],[],[],[]]
        importance_values=[[],[],[],[],[]]
        block_print = open(os.devnull, 'w')
        count=0
        user_profile_counts = []
        for user_id in userIDs:
            count+=1
            print(f"{count}/{len(userIDs)}........ {diversity_type}, {novelty_type}")
            importances = []
            for i in range(4):
                importances.append(random.randint(0, 100-sum(importances)))
            importances.append(100-sum(importances))
            random.shuffle(importances)
            importances = [importance / sum(importances) for importance in importances]
            sys.stdout = block_print
            support = predict_for_one_user(user_id,diversity_type, novelty_type,k,importances)
            sys.stdout = sys.__stdout__
            ranks.extend(list(range(1,k+1)))
            for i in range(len(supports_results)):
                importance_values[i].extend([importances[i]] * k)
                supports_results[i].extend(np.array(support)[:,i])
            user_profile_counts.extend([len(users_profiles[userIDs.index(user_id)])] * k)
        results_dfs =  pd.concat([pd.DataFrame({f"importance_{i}":importance_values[i],
                                                f"support_{i}":supports_results[i]                                               
                                                })                           
                                for i in range(len(supports_results))],
                                axis=1)
        ranks_and_types_df = pd.DataFrame({"rank":ranks,
                                            "user_profile_count": user_profile_counts,
                                            "diversity_type": [diversity_type] * len(ranks),
                                            "novelty_type": [novelty_type] * len(ranks),
                                            "novelty_type": [novelty_type] * len(ranks),
                                        })
        values_dfs.append(pd.concat([results_dfs, ranks_and_types_df], axis=1))
values_df = pd.concat(values_dfs)
values_df.to_csv("values.csv", index=False)
