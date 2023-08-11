import argparse
import os
import pickle
import random
import time
import types
import scipy
import copy
import mlflow
from support.calibration_support import calibration_support, set_params_calibration
from support.intra_list_distance_based_novelty_support import intra_list_distance_based_novelty_support

from support.maximal_distance_based_novelty_support import maximal_distance_based_novelty_support
from support.popularity_support import popularity_support
RUN_ID = os.environ[mlflow.tracking._RUN_ID_ENV_VAR] if mlflow.tracking._RUN_ID_ENV_VAR in os.environ else None

import glob

import numpy as np
import pandas as pd

from scipy.spatial.distance import squareform, pdist

from util import calculate_per_user_kl_divergence, calculate_per_user_errors

from mandate_allocation.exactly_proportional_fuzzy_dhondt import exactly_proportional_fuzzy_dhondt
from mandate_allocation.exactly_proportional_fuzzy_dhondt_2 import exactly_proportional_fuzzy_dhondt_2
from mandate_allocation.fai_strategy import fai_strategy
from mandate_allocation.probabilistic_fai_strategy import probabilistic_fai_strategy
from mandate_allocation.weighted_average_strategy import weighted_average_strategy
from mandate_allocation.sainte_lague_method import sainte_lague_method

from normalization.cdf import cdf
from normalization.standardization import standardization
from normalization.identity import identity
from normalization.robust_scaler import robust_scaler
from normalization.cdf_threshold_shift import cdf_threshold_shift

from support.rating_based_relevance_support import rating_based_relevance_support
from support.intra_list_diversity_support import intra_list_diversity_support
from support.maximal_diversity_support import maximal_diversity_support

from support.popularity_complement_support import popularity_complement_support

from support.binomial_diversity_support import set_params_bin_diversity, binomial_diversity_support
from support.popularity_based_log_novelty_support import popularity_based_log_novelty_support
import support.binomial_diversity_support

from mlflow import log_metric, log_param, log_artifacts, log_artifact, set_tracking_uri, set_experiment, start_run

from caserec.utils.process_data import ReadFile

from caserec.recommenders.rating_prediction.itemknn import ItemKNN
from caserec.recommenders.rating_prediction.matrixfactorization import MatrixFactorization
from caserec.recommenders.rating_prediction.base_rating_prediction import BaseRatingPrediction
from EASE.EASEModel import EASE

import pypyodbc as odbc
import pandas as pd



def get_supports(args, obj_weights, users_partial_lists, items, extended_rating_matrix,users_profiles , distance_matrix, users_viewed_item, k,\
                  num_users, user_index=None):
    default = np.repeat(np.zeros(len(items))[np.newaxis, :], users_partial_lists.shape[0], axis=0)
    rel_supps = div_supps = nov_supps = pop_supps = cal_supps =  default
    if (obj_weights[0]>0):
        rel_supps = rating_based_relevance_support(extended_rating_matrix)
    if (obj_weights[3]>0):
        pop_supps = popularity_support(users_viewed_item,users_partial_lists.shape[0], num_users)
    if (obj_weights[4]>0):
        cal_supps = calibration_support(users_partial_lists, items, user_index, k)
    if (obj_weights[1]>0):
        if (args.diversity == "intra_list_diversity"):
            div_supps = intra_list_diversity_support(users_partial_lists, items, distance_matrix, k)
        elif (args.diversity == "maximal_diversity"):
            div_supps = maximal_diversity_support(users_partial_lists, items, distance_matrix, k)
        elif (args.diversity == "binomial_diversity"):
            div_supps = binomial_diversity_support(users_partial_lists, items, user_index, k)            
    if (obj_weights[2]>0):    
        if (args.novelty == "popularity_complement"):
            nov_supps = popularity_complement_support(users_viewed_item,users_partial_lists.shape[0], num_users)
        elif (args.novelty == "popularity_based_log_novelty"):
            nov_supps = popularity_based_log_novelty_support(users_viewed_item,users_partial_lists.shape[0], num_users)
        elif (args.novelty == "maximal_distance_based_novelty"):
            nov_supps =  maximal_distance_based_novelty_support(users_profiles, items, distance_matrix)
        elif (args.novelty == "intra_list_distance_based_novelty"):
            nov_supps =  intra_list_distance_based_novelty_support(users_profiles, items, distance_matrix)

    return np.stack([rel_supps, div_supps, nov_supps, pop_supps, cal_supps])

def get_sparse_matrix_indices(matrix):
    major_dim, minor_dim = matrix.shape
    minor_indices = matrix.indices

    major_indices = np.empty(len(minor_indices), dtype=matrix.indices.dtype)
    scipy.sparse._sparsetools.expandptr(major_dim, matrix.indptr, major_indices)
    return zip(major_indices, minor_indices)


def save_cache(cache_path, cache):
    print(f"Saving cache to: {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)

def load_cache(cache_path):
    print(f"Loading cache from: {cache_path}")
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    return cache


# Parse movielens metadata
def parse_metadata(metadata_path, item_to_item_id, rating_matrix=None):
    all_genres, metadata_matrix, genre_to_genre_id = None, None, None
    if(support.binomial_diversity_support.metadata_matrix is None):
        metadata = dict()
        all_genres = set()
        movies_df = pd.read_csv(metadata_path)
        for index, row in movies_df.iterrows():
            genres = row["genres"].split("|")
            if genres[0]=="(no genres listed)":
                genres=[]
            all_genres.update(genres)
            metadata[int(row["movieId"])] = {
                "movie_name": row["title"],
                "genres": genres
            }
        genre_to_genre_id = {g:i for i, g in enumerate(all_genres)}
        metadata_matrix = np.zeros((len(item_to_item_id), len(all_genres)), dtype=np.int32)
        for movie, data in metadata.items():
            if movie not in item_to_item_id:
                continue
            item_id = item_to_item_id[movie]
            for g in data["genres"]:
                metadata_matrix[item_id, genre_to_genre_id[g]] = 1
    else:
        all_genres = support.binomial_diversity_support.all_genres
        metadata_matrix = support.binomial_diversity_support.metadata_matrix
        genre_to_genre_id = support.binomial_diversity_support.genre_to_genre_id

    metadata_distances = np.float32(squareform(pdist(metadata_matrix, "cosine")))
    metadata_distances[np.isnan(metadata_distances)] = 1.0
    #metadata_matrix = 1.0 - metadata_matrix
    genres_prob_all =np.zeros(len(all_genres))
    user_genre_prob = np.zeros((rating_matrix.shape[0], len(all_genres)))
    num_user_items=rating_matrix.getnnz( axis=1 )

    
    for i,j in get_sparse_matrix_indices(rating_matrix):
        user_genre_prob[i]+=metadata_matrix[j]
        genres_prob_all+=metadata_matrix[j]
    user_genre_prob=[user_genre_prob[i]/num_user_items[i] for i in range(user_genre_prob.shape[0])]

    num_all_items = rating_matrix.nnz
    genres_prob_all/=num_all_items
    


    return metadata_distances, user_genre_prob, genres_prob_all, metadata_matrix, genre_to_genre_id, all_genres


def get_EASE(args):
    ease = EASE(args.df)
    ease.fit(implicit=False)
    similarity_matrix = ease.compute_similarity(transpose=True)
    num_users = ease.X.shape[0]
    num_items = ease.B.shape[0]
    user_IDs = ease.user_enc.inverse_transform(list(range(num_users))).tolist()
    itemIDs = ease.item_enc.inverse_transform(list(range(num_items))).tolist()
#    unseen_items_mask = np.ones((num_users, num_items), dtype=np.bool8)
#    unseen_items_mask[ease.X.todense() > 0.0] = 0 # Mask out already seem items
    users_profiles = np.empty(num_users, dtype=object)
    users_profiles[...] = [[] for _ in range(users_profiles.shape[0])]
    for i,j in get_sparse_matrix_indices(ease.X):
        users_profiles[i].append(j)

    item_to_item_id = dict()
    item_id_to_item = dict()

    items = np.arange(num_items)
    users = np.arange(num_users)

    users_viewed_item = np.zeros_like(items, dtype=np.int32)
    Xt= ease.X.T.toarray()
    for idx, item in enumerate(itemIDs):
        item_to_item_id[item] = idx
        item_id_to_item[idx] = item
        users_viewed_item[idx] = sum([val> 0 for val in Xt[idx]])

    user_to_user_id = dict()
    user_id_to_user = dict()

    for idx, user in enumerate(user_IDs):
        user_to_user_id[user] = idx
        user_id_to_user[idx] = user

    extended_rating_matrix = ease.computeFullPredictionMatrix()
    extended_rating_matrix*=10
    extended_rating_matrix[extended_rating_matrix<0.0]=0
    """
    for u_id in range(extended_rating_matrix.shape[0]):
        for i_id in range(extended_rating_matrix.shape[1]):
            if ease.X[u_id, i_id] != 0.0:
                extended_rating_matrix[u_id, i_id] = ease.X[u_id, i_id]
            extended_rating_matrix[u_id, i_id]*=10
    """
    metadata_distance_matrix, user_genre_prob, genres_prob_all,metadata_matrix, genre_to_genre_id = None, None, None, None, None
    if hasattr(args, "metadata_path") and args.metadata_path:
        print(f"Parsing metadata from path: '{args.metadata_path}'")
        metadata_distance_matrix, user_genre_prob, genres_prob_all,metadata_matrix, genre_to_genre_id, all_genres = \
            parse_metadata(args.metadata_path, item_to_item_id, ease.X)

    return items, itemIDs, users, user_IDs, \
        users_viewed_item, item_to_item_id, \
        item_id_to_item, extended_rating_matrix, \
        users_profiles, similarity_matrix, \
        None, \
        metadata_distance_matrix, \
        user_id_to_user, user_to_user_id,\
        user_genre_prob, genres_prob_all,\
        metadata_matrix, genre_to_genre_id,\
        all_genres, ease.B




def get_baseline(args, baseline_factory):
    

    print(f"Calculating baseline '{baseline_factory.__name__}'")
    baseline = baseline_factory(args.train_path, sep=",")

    BaseRatingPrediction.compute(baseline)
    baseline.init_model()
    print(f"Training baseline '{baseline_factory.__name__}'")

    if hasattr(baseline, "fit"):
        baseline.fit()
    elif hasattr(baseline, "train_baselines"):
        baseline.train_baselines()
    else:
        assert False, "Fit/train_baselines not found for baseline"
    baseline.create_matrix()
    similarity_matrix = baseline.compute_similarity(transpose=True)

    train_set = baseline.train_set

    num_items = len(train_set['items'])
    num_users = len(train_set['users'])
    user_IDS = train_set['users']
    itemIDs = train_set['items']
    
    item_to_item_id = dict()
    item_id_to_item = dict()

    items = np.arange(num_items)
    users = np.arange(num_users)

    users_viewed_item = np.zeros_like(items, dtype=np.int32)

    for idx, item in enumerate(train_set['items']):
        item_to_item_id[item] = idx
        item_id_to_item[idx] = item
        users_viewed_item[idx] = len(train_set['users_viewed_item'][item])

    user_to_user_id = dict()
    user_id_to_user = dict()

    for idx, user in enumerate(train_set['users']):
        user_to_user_id[user] = idx
        user_id_to_user[idx] = user
    
    if baseline_factory == ItemKNN:
        print("Injecting into ItemKNN")
        def predict_score_wrapper(u_id, i_id):
            _, _, res = baseline.predict_scores(user_id_to_user[u_id], [item_id_to_item[i_id]])[0]
            return res
        setattr(baseline, "_predict_score", predict_score_wrapper)

    extended_rating_matrix = baseline.matrix.copy()
    for u_id in range(extended_rating_matrix.shape[0]):
        for i_id in range(extended_rating_matrix.shape[1]):
            if extended_rating_matrix[u_id, i_id] == 0.0:
                extended_rating_matrix[u_id, i_id] = baseline._predict_score(u_id, i_id)

    metadata_distance_matrix = None
    if args.metadata_path:
        print(f"Parsing metadata from path: '{args.metadata_path}'")
        metadata_distance_matrix = parse_metadata(args.metadata_path, item_to_item_id)
        
    

    return items, itemIDs, users, user_IDS, \
        users_viewed_item, item_to_item_id, \
        item_id_to_item, extended_rating_matrix, \
        similarity_matrix, \
        None, \
        metadata_distance_matrix, \
        user_id_to_user, user_to_user_id

def build_normalization(normalization_factory, shift):
    if shift:
        return normalization_factory(shift)
    else:
        return normalization_factory()

def compute_all_normalizations(args, normalization_factory, extended_rating_matrix, distance_matrix, \
                                           users_viewed_item, items, users):
    normalization = dict()
    shift = args.shift
    normalization["relevance"] = prepare_relevance_normalization(normalization_factory, extended_rating_matrix, shift)
    for diversity_type in ["intra_list_diversity","maximal_diversity","binomial_diversity"]:
        normalization[diversity_type] = prepare_diversity_normalization(diversity_type, normalization_factory, \
                                                                        distance_matrix, shift, items, users)
    for novelty_type in ["popularity_complement","maximal_distance_based_novelty","intra_list_distance_based_novelty",\
                         "popularity_based_log_novelty"]:
        normalization[novelty_type] = prepare_novelty_normalization(novelty_type, normalization_factory, extended_rating_matrix,\
                                                                    distance_matrix, users_viewed_item, shift, items)
    normalization["popularity"] = prepare_popularity_normalization(normalization_factory, users_viewed_item, shift,\
                                                                     extended_rating_matrix.shape[0])
    normalization["calibration"] = prepare_calibration_normalization( normalization_factory, distance_matrix, shift, items, users)
    return normalization


def prepare_relevance_normalization(normalization_factory, rating_matrix, shift):

    relevance_data_points = rating_matrix.T

    norm_relevance = build_normalization(normalization_factory, shift)
    norm_relevance.train(relevance_data_points)

    return norm_relevance

def prepare_diversity_normalization(diversity_type, normalization_factory, distance_matrix, shift, items, users):
    users_partial_lists= np.full((1,15), -1, dtype=np.int32)
    diversity_data_points=[]
    user_index = random.choice(list(range(len(users))))
    for i in range(15):        
        if(diversity_type=="binomial_diversity"):
            diversity_data_points.extend(binomial_diversity_support(users_partial_lists, items, user_index, i+1))
        elif (diversity_type=="maximal_diversity"):
            diversity_data_points.extend(maximal_diversity_support(users_partial_lists,items, distance_matrix, i+1))
        elif (diversity_type=="intra_list_diversity"):
            diversity_data_points.extend(intra_list_diversity_support(users_partial_lists,items, distance_matrix, i+1))
        users_partial_lists[0, i] = np.random.choice([np.argmax(diversity_data_points[i]),np.random.choice(list(range(len(items))))],\
                                                    p=[0.2, 0.8])
    diversity_data_points=np.expand_dims(random.choices(np.array(diversity_data_points).flatten(), k = len(items)),axis=1)
    norm_diversity = build_normalization(normalization_factory, shift)
    norm_diversity.train(diversity_data_points)
    return norm_diversity

def prepare_novelty_normalization(novelty_type, normalization_factory, rating_matrix, distance_matrix, users_viewed_item, shift, items):
    num_users = rating_matrix.shape[0] 
    novelty_data_points=[]
    if(novelty_type=="popularity_complement"):        
        novelty_data_points = np.expand_dims(1.0 - users_viewed_item / num_users, axis=1)
    elif(novelty_type=="popularity_based_log_novelty"):
        novelty_data_points = np.expand_dims( - np.log2(users_viewed_item / num_users), axis=1)
    else:
        novelty_data_points= [] 
        for i in range(10):
            user_list = np.expand_dims(random.choices(items, k = i*5 + 1), axis=0)
            if(novelty_type=="maximal_distance_based_novelty"):
                novelty_data_points.extend(maximal_distance_based_novelty_support(user_list, items, distance_matrix))
            elif (novelty_type=="intra_list_distance_based_novelty"):
                novelty_data_points.extend(intra_list_distance_based_novelty_support(user_list, items, distance_matrix))
        novelty_data_points=np.expand_dims(random.choices(np.array(novelty_data_points).flatten(), k = len(items)),axis=1)

    norm_novelty = build_normalization(normalization_factory, shift)
    norm_novelty.train(novelty_data_points)

    return norm_novelty


def prepare_popularity_normalization(normalization_factory, users_viewed_item, shift, num_users):
    popularity_data_points =  np.expand_dims(users_viewed_item / num_users, axis=1)
    norm_popularity = build_normalization(normalization_factory, shift)
    norm_popularity.train(popularity_data_points)
    return norm_popularity

def prepare_calibration_normalization( normalization_factory, distance_matrix, shift, items, users):
    users_partial_lists= np.full((1,15), -1, dtype=np.int32)
    calibration_data_points=[]
    user_index = random.choice(list(range(len(users))))
    for i in range(15):        
        calibration_data_points.extend(calibration_support(users_partial_lists, items, user_index, i+1))
        users_partial_lists[0, i] = np.random.choice([np.argmax(calibration_data_points[i]),np.random.choice(list(range(len(items))))],\
                                                    p=[0.2, 0.8])
    calibration_data_points=np.expand_dims(random.choices(np.array(calibration_data_points).flatten(), k = len(items)),axis=1)
    norm_calibration = build_normalization(normalization_factory, shift)
    norm_calibration.train(calibration_data_points)
    return norm_calibration

def custom_evaluate_voting(top_k, rating_matrix, distance_matrix, users_viewed_item, normalizations, obj_weights, discount_sequences):
    start_time = time.perf_counter()
    
    [mer_norm, div_norm, nov_norm] = normalizations

    num_users = top_k.shape[0]    

    normalized_mer = 0.0
    normalized_diversity = 0.0
    normalized_novelty = 0.0
    
    normalized_per_user_mer = []
    normalized_per_user_diversity = []
    normalized_per_user_novelty = []

    normalized_per_user_mer_matrix = mer_norm(np.sum(np.take_along_axis(rating_matrix, top_k, axis=1) * discount_sequences[0], axis=1, keepdims=True).T / discount_sequences[0].sum(), ignore_shift=False).T
    
    total_mer = 0.0
    total_novelty = 0.0
    total_diversity = 0.0
    
    per_user_mer = []
    per_user_diversity = []
    per_user_novelty = []
    n = 0
    for user_id, user_ranking in enumerate(top_k):
        
        relevance = (rating_matrix[user_id][user_ranking] * discount_sequences[0]).sum()
        novelty = ((1.0 - users_viewed_item[user_ranking] / rating_matrix.shape[0]) * discount_sequences[2]).sum()
        div_discount = np.repeat(np.expand_dims(discount_sequences[1], axis=0).T, user_ranking.size, axis=1)
        diversity = (distance_matrix[np.ix_(user_ranking, user_ranking)] * div_discount).sum() / user_ranking.size

        # Per user MER
        normalized_per_user_mer.append(normalized_per_user_mer_matrix[user_id].item())
        normalized_mer += normalized_per_user_mer[-1]
        
        # Per user Diversity
        ranking_distances = distance_matrix[np.ix_(user_ranking, user_ranking)] * div_discount
        triu_indices = np.triu_indices(user_ranking.size, k=1)
        ranking_distances_mean = ranking_distances[triu_indices].sum() / div_discount[triu_indices].sum()
        normalized_ranking_distances_mean = div_norm([[ranking_distances_mean]], ignore_shift=False)
        normalized_per_user_diversity.append(normalized_ranking_distances_mean.item())
        normalized_diversity += normalized_per_user_diversity[-1]

        # Per user novelty
        normalized_per_user_novelty.append(nov_norm(((1.0 - users_viewed_item[user_ranking] / num_users) * discount_sequences[2]).sum().reshape(-1, 1) / discount_sequences[2].sum(), ignore_shift=False).item())
        normalized_novelty += normalized_per_user_novelty[-1]
        
        per_user_mer.append(relevance)
        per_user_diversity.append(diversity)
        per_user_novelty.append(novelty)

        total_mer += relevance
        total_diversity += diversity
        total_novelty += novelty
        n += 1

    total_mer = total_mer / n
    total_diversity = total_diversity / n
    total_novelty = total_novelty / n

    normalized_mer = normalized_mer / n
    normalized_diversity = normalized_diversity / n
    normalized_novelty = normalized_novelty / n

    per_user_kl_divergence = calculate_per_user_kl_divergence(normalized_per_user_mer, normalized_per_user_diversity, normalized_per_user_novelty, obj_weights)
    per_user_mean_absolute_errors, per_user_errors = calculate_per_user_errors(normalized_per_user_mer, normalized_per_user_diversity, normalized_per_user_novelty, obj_weights)

    print(f"per_user_kl_divergence: {per_user_kl_divergence}")
    print(f"per_user_mean_absolute_errors: {per_user_mean_absolute_errors}")
    print(f"per_user_errors: {per_user_errors}")

    print("####################")
    print(f"MEAN ESTIMATED RATING: {total_mer}")
    print(f"DIVERSITY2: {total_diversity}")
    print(f"NOVELTY2: {total_novelty}")
    print("--------------------")
    log_metric("raw_mer", total_mer)
    log_metric("raw_diversity", total_diversity)
    log_metric("raw_novelty", total_novelty)

    print(f"Normalized MER: {normalized_mer}")
    print(f"Normalized DIVERSITY2: {normalized_diversity}")
    print(f"Normalized NOVELTY2: {normalized_novelty}")
    print("--------------------")
    log_metric("normalized_mer", normalized_mer)
    log_metric("normalized_diversity", normalized_diversity)
    log_metric("normalized_novelty", normalized_novelty)

    # Print sum-to-1 results
    s = normalized_mer + normalized_diversity + normalized_novelty
    print(f"Sum-To-1 Normalized MER: {normalized_mer / s}")
    print(f"Sum-To-1 Normalized DIVERSITY2: {normalized_diversity / s}")
    print(f"Sum-To-1 Normalized NOVELTY2: {normalized_novelty / s}")
    print("--------------------")
    log_metric("normalized_sum_to_one_mer", normalized_mer / s)
    log_metric("normalized_sum_to_one_diversity", normalized_diversity / s)
    log_metric("normalized_sum_to_one_novelty", normalized_novelty / s)

    mean_kl_divergence = np.mean(per_user_kl_divergence)
    mean_absolute_error = np.mean(per_user_mean_absolute_errors)
    mean_error = np.mean(per_user_errors)

    print(f"mean_kl_divergence: {mean_kl_divergence}")
    print(f"mean_absolute_error: {mean_absolute_error}")
    print(f"mean_error: {mean_error}")
    print("####################")
    log_metric("mean_kl_divergence", mean_kl_divergence)
    log_metric("mean_absolute_error", mean_absolute_error)
    log_metric("mean_error", mean_error)

    print(f"Evaluation took: {time.perf_counter() - start_time}")
    return {
        "mer": total_mer,
        "diversity": total_diversity,
        "novelty": total_novelty,
        "per-user-mer": per_user_mer,
        "per-user-diversity": per_user_diversity,
        "per-user-novelty": per_user_novelty,
        "normalized-mer": normalized_mer,
        "normalized-diversity": normalized_diversity,
        "normalized-novelty": normalized_novelty,
        "normalized-per-user-mer": normalized_per_user_mer,
        "normalized-per-user-diversity": normalized_per_user_diversity,
        "normalized-per-user-novelty": normalized_per_user_novelty,
        "mean-kl-divergence": mean_kl_divergence,
        "mean-absolute-error": mean_absolute_error,
        "mean-error": mean_error,
        "sum-to-1-normalized-mer": normalized_mer / s,
        "sum-to-1-normalized-diversity": normalized_diversity / s,
        "sum-to-1-normalized-novelty": normalized_novelty / s,
        "per-user-kl-divergence": per_user_kl_divergence,
        "per-user-mean-absolute-errors": per_user_mean_absolute_errors,
        "per-user-errors": per_user_errors
    }

def predict_for_user(user, user_index, items, extended_rating_matrix, users_profiles, distance_matrix,\
                      users_viewed_item, normalizations, mask, algorithm_factory, args, obj_weights, num_users):
    start_time = time.perf_counter()
    mandate_allocation = algorithm_factory(obj_weights, args.masking_value)
    users_partial_lists= np.full((1,args.k), -1, dtype=np.int32)
    supports_partial_lists = []
    for i in range(args.k):
        iter_start_time = time.perf_counter()
        print(f"Predicting for i: {i + 1} out of: {args.k}")
        # Calculate support values
        #supports = get_supports(users_partial_lists, items, extended_rating_matrix[user_index:user_index+1],\
        #                         distance_matrix, users_viewed_item, k=i+1, num_users=num_users)
        supports = get_supports(args, obj_weights, users_partial_lists, items, extended_rating_matrix[user_index:user_index+1],\
                                users_profiles[user_index:user_index+1],distance_matrix, users_viewed_item, k=i+1, \
                                    num_users=num_users, user_index=user_index)
        
        # Normalize the supports
        assert supports.shape[0] == 5, "expecting 5 objectives, if updated, update code below"

        # Discount sequence can be [1] * K, i.e. 1.0 at every position
        # 0 -> relevance, 1 -> diversity, 2 -> novelty
        supports[0, :, :] = normalizations["relevance"](supports[0].T, userID=user_index).T * args.discount_sequences[0][i]
        supports[1, :, :] = normalizations[args.diversity](supports[1].reshape(-1, 1)).reshape((supports.shape[1], -1)) \
            * args.discount_sequences[1][i] 
        supports[2, :, :] = normalizations[args.novelty](supports[2].reshape(-1, 1)).reshape((supports.shape[1], -1)) \
            * args.discount_sequences[2][i]
        supports[3, :, :] = normalizations["popularity"](supports[3].reshape(-1, 1)).reshape((supports.shape[1], -1)) \
                    * args.discount_sequences[3][i] 
        supports[4, :, :] = normalizations["calibration"](supports[4].reshape(-1, 1)).reshape((supports.shape[1], -1)) \
                    * args.discount_sequences[4][i] 
        
        # Mask out the already recommended items
        np.put_along_axis(mask, users_partial_lists[:, :i], 0, 1)

        # Get the per-user top-k recommendations
        users_partial_lists[:, i] = mandate_allocation(mask, supports)

        # For explanations, get supports of each item w.r.t. objectives
        item_supports = np.squeeze(supports[:, 0, users_partial_lists[0, i]])
        item_supports = [item_support * 100 for item_support in item_supports]
        supports_partial_lists.append(item_supports)
        print(args.diversity + ", " + args.novelty)
        print(item_supports)
        print(f"i: {i + 1} done, took: {time.perf_counter() - iter_start_time}")
    print(f"Prediction done, took: {time.perf_counter() - start_time}")
    print("average\n")
    print(np.average(np.array(supports_partial_lists), axis=0))
    return users_partial_lists, supports_partial_lists

def main(args):
    for arg_name in dir(args):
        if arg_name[0] != '_':
            arg_value = getattr(args, arg_name)
            print(f"\t{arg_name}={arg_value}")

    if not args.normalization:
        print(f"Using Identity normalization")
        normalization_factory = identity
    else:
        print(f"Using {args.normalization} normalization")
        normalization_factory = globals()[args.normalization]

    algorithm_factory = globals()[args.algorithm]
    print(f"Using '{args.algorithm}' algorithm")
    if((globals()[args.baseline]) is EASE):
        items,itemIDs, users, userIDs, users_viewed_item, item_to_item_id, item_id_to_item, extended_rating_matrix,users_profiles,\
              similarity_matrix, test_set_users_start_index, metadata_distance_matrix, user_id_to_user, user_to_user_id, \
                user_genre_prob, genres_prob_all, metadata_matrix, genre_to_genre_id, all_genres, ease_B = get_EASE(args)
        set_params_bin_diversity(user_genre_prob, genres_prob_all, metadata_matrix, genre_to_genre_id, all_genres)
        set_params_calibration(user_genre_prob, metadata_matrix)

    else:
        items, itemIDs, users, userIDs, users_viewed_item, item_to_item_id, item_id_to_item, extended_rating_matrix, similarity_matrix, \
        test_set_users_start_index, metadata_distance_matrix, user_id_to_user, user_to_user_id = get_baseline(args, globals()[args.baseline])
    if args.type_of_items_distance == "cb":
        print("Using content based diversity")
        assert args.metadata_path, "Metadata path must be specified when using cb diversity"
        distance_matrix = metadata_distance_matrix
    elif args.type_of_items_distance == "cf":
        print("Using collaborative diversity")
        distance_matrix = 1.0 - similarity_matrix
    else:
        assert False, f"Unknown diversity: {args.type_of_items_distance}"
    #extended_rating_matrix = (extended_rating_matrix - 1.0) / 4.0

    # Prepare normalizations
    

    num_users = users.size
    
    obj_weights = args.weights
    obj_weights /= obj_weights.sum()

    start_time = time.perf_counter()

    mandate_allocation = algorithm_factory(obj_weights, args.masking_value) # Algorithm implementation (./mandate_allocation/*.py)
 #   users_partial_lists= np.full((num_users, args.k), -1, dtype=np.int32) # Output tensor (shape=[NUM_USERS, K])


    # extended_rating_matrix -> same as rating matrix, but missing entries are replaced by baseline predictions (shape=[NUM_USERS, NUM_ITEMS])
    # distance_matrix -> distance between every pair of items (shape=[NUM_ITEMS, NUM_ITEMS])
    # users_vievew_item => np.array, for each item we count number of users who saw it (based on training data)

    # List of normalizations (relevance normalization, diversity normalization, novelty normalization)
    # args.shift can be 0
    # normalization_factory is either cdf or standardization, defined in ./normalization/cdf.py, ./normalization/standardization.py
    start_time = time.perf_counter()

    normalizations = compute_all_normalizations(args, normalization_factory, extended_rating_matrix, distance_matrix, \
                                           users_viewed_item, items, users)
    print(f"Preparing normalizations took: {time.perf_counter() - start_time}")

    print(f"### Whole prediction took: {time.perf_counter() - start_time} ###")

    return extended_rating_matrix, users_profiles,users_viewed_item, distance_matrix, items,itemIDs, users, userIDs, \
        algorithm_factory, normalizations, args, ease_B

def get_ratings(connectionstring):
    conn = odbc.connect(connectionstring)
    df = pd.read_sql_query('SELECT  * FROM ratings', conn)
    df = df.drop(columns=["id","date"])
    return df

def get_ratings_of_user(connectionstring, userID):
    conn = odbc.connect(connectionstring)
    df = pd.read_sql_query(f'SELECT  * FROM ratings where userid = {userID}', conn)
    df = df.drop(columns=["id","date"])
    return df

def save_ratings_from_db_to_file():
    df = get_ratings()
    df.to_csv("ratings.csv", index=False,  header=False)
    




def init():
    args=types.SimpleNamespace()
    args.k = 10
    args.train_path="ratings.csv"
    args.seed=42
    args.weights="0.3,0.3,0.3"
    args.normalization="cdf_threshold_shift"
    args.algorithm="exactly_proportional_fuzzy_dhondt_2"
    args.novelty = "intra_list_distance_based_novelty"
    args.diversity="binomial_diversity"
    args.masking_value=-1e6
    args.baseline="EASE"
    args.metadata_path="movies.csv"
    args.type_of_items_distance="cf"
    args.shift=0.0
    args.artifact_dir=None
    args.output_path_prefix=None
    args.discounts="1,1,1,1,1"
    args.weights = np.fromiter(map(float, args.weights.split(",")), dtype=np.float32)
    args.discounts = [float(d) for d in args.discounts.split(",")]
    args.discount_sequences = np.stack([np.geomspace(start=1.0,stop=d**args.k, num=args.k, endpoint=False) for d in args.discounts], axis=0)


    DriverName = "SQL Server"
    ServerName =  "np:\\\\.\\pipe\LOCALDB#53699C9C\\tsql\\query"
    DatabaseName = "aspnet-53bc9b9d-9d6a-45d4-8429-2a2761773502"
    args.connectionstring=f"""DRIVER={{{DriverName}}};
        SERVER={ServerName};
        DATABASE={DatabaseName};
        Trusted_Connection=yes;"""
    
    """
    if not args.artifact_dir:
        print("Artifact directory is not specified, trying to set it")
        if not RUN_ID:
            print("Not inside mlflow's run, leaving artifact directory empty")
        else:
            print(f"Inside mlflow's run {RUN_ID} setting artifact directory")
            if args.output_path_prefix:
                args.artifact_dir = os.path.join(args.output_path_prefix, RUN_ID)
                print(f"Set artifact directory to: {args.artifact_dir}")
            else:
                print("Output path prefix is not set, skipping setting of artifact directory")
    """
    args.df = get_ratings(args.connectionstring)
    random.seed(args.seed)
    return main(args)

if __name__ == "__main__":
    init()