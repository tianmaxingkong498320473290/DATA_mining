import numpy as np
from scipy import linalg
from itertools import combinations

class OptimalSubsetRegression:
    def __init__(self, x, y, index, feature_names, K):
        self.x = x
        self.y = y
        self.index = index
        self.feature_names = feature_names
        self.K = K
        self.xtrain, self.xtest = x[index], x[~index]
        self.ytrain, self.ytest = y[index], y[~index]
        self.n, self.p = self.xtrain.shape
        self.feature_combinations = self._generate_feature_combinations()
        
    def _generate_feature_combinations(self):
        return [list(comb) for r in range(1, self.p + 1) for comb in combinations(range(self.p), r)]
    
    def _ordinary_least_squares(self, x, y):
        beta_1, _ = linalg.lapack.dpotrs(linalg.cholesky(np.dot(x.T, x)), np.dot(x.T,y))
        beta_0 = y.mean() - np.dot(beta_1, x.mean(axis=0))
        return beta_0, beta_1
    
    def _train_model(self, features):
        x_centered = self.xtrain[:, features] - self.xtrain[:, features].mean(axis=0)
        y_centered = self.ytrain - self.ytrain.mean()
        return self._ordinary_least_squares(x_centered, y_centered)
    
    def train(self):
        self.models = [self._train_model(features) for features in self.feature_combinations]
        
    def predict_error(self, x, y, model):
        b0, b1 = model
        prediction = b0 + np.dot(x, b1)
        error = y - prediction
        return np.dot(error, error)
    
    def evaluate(self):
        self.test_errors = [self.predict_error(self.xtest[:, features], self.ytest, model)
                            for features, model in zip(self.feature_combinations, self.models)]
        self.training_errors = [self.predict_error(self.xtrain[:, features], self.ytrain, model)
                                for features, model in zip(self.feature_combinations, self.models)]
    
    def cross_validate(self):
        indices = np.array_split(np.random.permutation(self.n), self.K)
        self.cross_validation_errors = []

        for index in indices:
            x_i = self.xtrain[index, :]
            y_i = self.ytrain[index]
            cv_errors_i = [self.predict_error(x_i[:, features], y_i, model)
                            for features, model in zip(self.feature_combinations, self.models)]
            self.cross_validation_errors.append(np.mean(cv_errors_i))

        self.avg_cross_validation_error = np.mean(self.cross_validation_errors)
        
    def best_model(self):
        best_index = np.argmin(self.cross_validation_errors)
        best_b_0, best_b_1 = self.models[best_index]
        best_features = self.feature_combinations[best_index]
        selected_names = [self.feature_names[i] for i in best_features]
        return best_b_0, best_b_1, selected_names