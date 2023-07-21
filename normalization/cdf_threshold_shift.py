import numpy as np
import copy 

from sklearn.preprocessing import QuantileTransformer

class cdf_threshold_shift:
    def __init__(self, shift=0.0):
        self.transformer = QuantileTransformer()
        self.threshold = None
        self.shift = shift

    def __call__(self, supports, ignore_shift=False, userID=None):
        if ignore_shift:
            return self.transformer.transform(supports)
        
        if userID is not None:
            part_transformer = copy.deepcopy(self.transformer)
            part_transformer.quantiles_=self.transformer.quantiles_[:,userID].reshape(self.transformer.quantiles_.shape[0],1)
            part_transformer.n_features_in_=1

            # supports have shape [num_users, num_data_points] or [num_data_points]
            transformed_supports = part_transformer.transform(supports)
            # Shift only values that are below threshold
            transformed_supports[transformed_supports < self.threshold] += self.shift
            return transformed_supports
        # supports have shape [num_users, num_data_points] or [num_data_points]
        transformed_supports = self.transformer.transform(supports)
        # Shift only values that are below threshold
        transformed_supports[transformed_supports < self.threshold] += self.shift
        return transformed_supports

    def train(self, data_points):
        transformed_data_points = self.transformer.fit_transform(data_points)
        self.threshold = np.percentile(np.unique(transformed_data_points), 50)
        print(f"@@@ Threshold: {self.threshold}")