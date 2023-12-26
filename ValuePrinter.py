from datetime import datetime
from app import create_new_user, get_mask, set_rating_matrices_row
import main
import copy
import numpy as np
import random
import pandas as pd
import sys
import os
import app

from app import init
from support.binomial_diversity_support import get_user_genre_prob, recompute_user_genres_prob
import support.calibration_support as calibration_support



def get_first_k_rated(user_id, userindex, k):
    global simulated_users_ratings
    neg_rated = simulated_users_ratings[user_id]
    neg_rated = neg_rated.head(k)
    pos_rated = neg_rated[neg_rated["ratingscore"] > 5]
    pos_scores = list(pos_rated["ratingscore"].apply(lambda x: x/10.0 if x>5 else 0))
    neg_scores = list(neg_rated["ratingscore"].apply(lambda x: (x - 5) * 2 /10.0))
    pos_rated = list(pos_rated["itemid"])
    pos_rated = list(map(app.itemIDs.index, pos_rated))
    neg_rated = list(neg_rated["itemid"])
    neg_rated = list(map(app.itemIDs.index, neg_rated))
    rated = pos_rated
    scores = pos_scores
    if (app.args.relevance == "also_negative"):
        rated = neg_rated
        scores = neg_scores
    app.pos_users_profiles[userindex] = pos_rated
    app.neg_users_profiles[userindex] = neg_rated
    return rated, scores, pos_rated

profile_counts = np.load("users_profile_counts_randomly.npy")
simulated_userIDs = [int(record.split(';')[0]) for record in profile_counts][:1000]
profile_counts = [int(record.split(';')[1]) for record in profile_counts]
init(True)
#simulated_userIDs = random.sample(list(app.userIDs), k=3)
simulated_users_ratings = {}
for id in simulated_userIDs:
    simulated_users_ratings[id] = main.get_ratings_of_user(app.args.connectionstring, id,\
                                        only_positive=False, order=True)
random_lengths = []
k = 15
profile_counts_index=0
values_dfs = []
for app.args.diversity_type in ["binomial_diversity","intra_list_diversity","maximal_diversity"]:
    for app.args.novelty_type in ["popularity_complement","maximal_distance_based_novelty","intra_list_distance_based_novelty"]:
        for app.args.relevance_type in ["only_positive", "also_negative"]:
            for app.args.popularity_type in ["num_of_ratings","avg_ratings"]:
                ranks=[]
                supports_results=[[],[],[],[],[]]
                importance_values=[[],[],[],[],[]]
                block_print = open(os.devnull, 'w')
                count=0
                user_profile_counts = []
                for user_id in simulated_userIDs:
                    count+=1
                    new_user_id = max(app.userIDs) + count
                    create_new_user(new_user_id)
                    userindex =app.userIDs.index(new_user_id) 
                    l = profile_counts[profile_counts_index]#random.randint(3, 50)
                    profile_counts_index+=1
                    #random_lengths.append(f"{user_id};{l}")
                    rated, scores, pos_rated = get_first_k_rated(user_id, userindex, l) 
                    user_id = new_user_id
                    set_rating_matrices_row(rated, scores, app.args, userindex)
                    recompute_user_genres_prob(userindex, pos_rated) 
                    calibration_support.set_user_genre_prob(userindex, get_user_genre_prob()[userindex])
                    user_mask = get_mask([], rated, []) 
                    print(f"{count}/{len(simulated_userIDs)}........ {app.args.diversity_type}, {app.args.novelty_type}, {app.args.popularity_type}, {app.args.relevance_type}")
                    importances = []
                    for i in range(4):
                        importances.append(random.randint(0, 100-sum(importances)))
                    importances.append(100-sum(importances))
                    random.shuffle(importances)
                    importances = [importance / sum(importances) for importance in importances]
                    sys.stdout = block_print
                    result, support = main.predict_for_user(app.users[userindex], userindex,app.items,app.extended_rating_matrix,\
                                            app.neg_extended_rating_matrix,app.pos_users_profiles,app.neg_users_profiles,\
                                                   app.distance_matrix,\
                                           app.users_viewed_item,app.normalizations, np.array(user_mask),\
                                           app.algorithm_factory, app.args, np.array(importances), len(app.users),\
                                            [])
                    sys.stdout = sys.__stdout__
                    ranks.extend(list(range(1,k+1)))
                    for i in range(len(supports_results)):
                        importance_values[i].extend([importances[i]] * k)
                        supports_results[i].extend(np.array(support)[:,i])
                    user_profile_counts.extend([len(app.neg_users_profiles[app.userIDs.index(user_id)])] * k)
                results_dfs =  pd.concat([pd.DataFrame({f"importance_{i}":importance_values[i],
                                                        f"support_{i}":supports_results[i]                                               
                                                        })                           
                                        for i in range(len(supports_results))],
                                        axis=1)
                ranks_and_types_df = pd.DataFrame({"rank":ranks,
                                                    "user_profile_count": user_profile_counts,
                                                    "diversity_type": [app.args.diversity_type] * len(ranks),
                                                    "novelty_type": [app.args.novelty_type] * len(ranks),
                                                    "popularity_type": [app.args.popularity_type] * len(ranks),
                                                    "relevance_type": [app.args.relevance_type] * len(ranks)
                                                })
                values_dfs.append(pd.concat([results_dfs, ranks_and_types_df], axis=1))
                print(datetime.now)
                values_df = pd.concat(values_dfs)
                values_df.to_csv("values_21_12_old_temp.csv", index=False)
values_df = pd.concat(values_dfs)
values_df.to_csv("values_21_12_old.csv", index=False)
#np.save("users_profile_counts_randomly",np.array(random_lengths))
