from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool, cpu_count
from AutoencoderRS import AutoencoderRS
from itertools import starmap

class EASE(AutoencoderRS):
    """
    https://arxiv.org/pdf/1905.03375.pdf
    https://github.com/Darel13712/ease_rec
    """
    
    def computeB(self,lambda_):
        """
        Parameters
        ----------
        lambda_ : float
            L2-regulization parameter

        Returns
        -------
        np.array
            Matrix B computed from X (closed solution from paper)
        """
        G = self.X.T.dot(self.X).toarray()
        diagIndices = np.diag_indices(G.shape[0])
        G[diagIndices] += lambda_
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[diagIndices] = 0
        return B
    
    
    def fit(self, lambda_: float = 1000, implicit=True):
        """
        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe with columns user_id, item_id and (RatingScore)
        lambda_ : float, optional
            L2-regulization parameter, by default 1000
        implicit : bool, optional
            True - using 1 and 0, False using value from column RatingScore, by default True
        """
        df = self.df
        df['ratingscore'] = df['ratingscore'].apply(lambda x: x if x>5 else 0)
        users, items = self._get_users_and_items(df)
        values = (
            np.ones(df.shape[0])
            if implicit
            else df['ratingscore'] / 10.0
        )
        self.X = csr_matrix((values, (users, items)))   
        self.B = self.computeB(lambda_)

        
    def computePrediction(self, user):
        """
        Parameters
        ----------
        user : int
            index of user

        Returns
        -------
        np.array
            prediction row (=item scores) for one user
        """
        return self.X[user, :].dot(self.B)
    

    def computeFullPredictionMatrix(self):
        """
        Returns
        -------
        np.array
            Whole matrix users*items with predictions
        """
        return self.X.dot(self.B)
        