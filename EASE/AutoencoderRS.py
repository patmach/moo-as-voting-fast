from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import squareform, pdist
from itertools import starmap

class AutoencoderRS():


    def __init__(self, df):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()
        self.df = df

    def _get_users_and_items(self, df):
        """
        Transform ID of users and items to indices of matrix

        Parameters
        ----------
        df : pd.DataFrame
            Train dataset with columns user_id and item_id

        Returns
        -------
        _type_
            Transformed IDs (=indices) of users and items
        """
        users = self.user_enc.fit_transform(df.loc[:, 'userid'])
        items = self.item_enc.fit_transform(df.loc[:, 'itemid'])
        return users, items

    def computePrediction(self, user):
        """
        Abstract, every algorithm computes it differently
        Parameters
        ----------
        user : int
            index of user

        Returns
        -------
        np.array
            prediction row (=item scores) for one user
        
        """
        raise NotImplementedError()
    
    def computeFullPredictionMatrix(self):
        """
        Abstract, every algorithm computes it differently
        Returns
        -------
        np.array
            Whole matrix users*items with predictions
        """
        raise NotImplementedError()

    
    def predict(self, train, users, items, k, only_items_without_transaction = False, get_all = False):
        """
        Predicts scores for users and returns dataset with top k recommendations for each
        Parameters
        ----------
        train : pd.DataFrame
            The same dataframe that was used as parameter of fit
        users : _type_
            Users that the method returns recommendations to
        items : _type_
            Candidates for recommendation
        k : int
            How many recommendations should be returned to user
        only_items_without_transaction : bool, optional
            Will I recommend only items where users havent made transaction before, by default False
        get_all : bool, optional
            Ignoring parameter k and returns all possible candidates, by default False

        Returns
        -------
        pd.DataFrame
            DataFrame in format: [user_id, item_id, score, rank(for user)]
        """
        items = self.item_enc.transform(items)
        dd = train.loc[train.user_id.isin(users)]
        dd['cm'] = self.item_enc.transform(dd.item_id)
        dd['cc'] = self.user_enc.transform(dd.user_id)
        g = dd.groupby('cc')
        user_preds = starmap(
                self.predict_for_user,
                [(user, group, self.computePrediction(user), items, k, only_items_without_transaction, get_all)\
                  for user, group in g],)
        df = pd.concat(user_preds)
        df['item_id'] = self.item_enc.inverse_transform(df['item_id'])
        df['user_id'] = self.user_enc.inverse_transform(df['user_id'])
        return df


    def predict_for_user(self,user, group,  pred, items, k, only_items_without_transaction = False, get_all = False):
        """Computes predictions for one user

        Parameters
        ----------
        user : int
            Index of user
        group : _type_
            group of item indices where user has a transaction
        pred : _type_
            prediction = score of items for user
        items : _type_
            candidates - items that are possible for recommendations
        k : int
            How many recommendations should be returned to user
        only_items_without_transaction : bool, optional
            Will I recommend only items where users havent made transaction before, by default False
        get_all : bool, optional
            Ignoring parameter k and returns all possible candidates, by default False

        Returns
        -------
        pd.DataFrame
            Results of prediction for one user in dataframe with columns user_id, item_id, score, rank
        """
        candidates = items
        if(only_items_without_transaction):
            with_transaction = set(group['cm'])
            candidates = [item for item in items if item not in with_transaction]
        if(get_all):
            k = len(candidates)
        pred = np.take(pred, candidates)
        res = np.argpartition(pred, -k)[-k:]
        r = pd.DataFrame(
            {
                "user_id": [user] * len(res),
                "item_id": np.take(candidates, res),
                "score": np.take(pred, res),
            }
        ).sort_values('score', ascending=False)
        r["rank"]=list(range(1, len(res)+1))
        return r
    
    def compute_similarity(self, transpose=False, similarity_metric='cosine'):
        """
        Method to compute a similarity matrix from original df_matrix

        :param transpose: If True, calculate the similarity in a transpose matrix
        :type transpose: bool, default False

        """

        # Calculate distance matrix
        if transpose:
            similarity_matrix = np.float32(squareform(pdist(self.X.todense().T, similarity_metric)))
        else:
            similarity_matrix = np.float32(squareform(pdist(self.X.todense(), similarity_metric)))

        # Remove NaNs
        similarity_matrix[np.isnan(similarity_matrix)] = 1.0
        # transform distances in similarities. Values in matrix range from 0-1
        similarity_matrix = (similarity_matrix.max() - similarity_matrix) / similarity_matrix.max()

        return similarity_matrix
