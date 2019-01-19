### Environment Preparation
import numpy as np
import pandas as pd
import random

### Main Class
class DAT(object):

    def __init__(self,data):
        """
        data: pandas DataFrame
        """
        self.mu = pd.Series(data.mean(axis=1, skipna=True)) # sample mean for each item
        self.mu = np.array(self.mu) # mu: ndarray
        self.data = np.array(data) # data: ndarray
        self.flag = pd.DataFrame(np.isnan(self.data)) # missing value flags

    def CF(self, m = 10, sigma = 0.01, eta = 0.01, lmbda = 0.0, epoch = 100, N = 1, d = 2):
        """
        Collaborative Filtering (sensitive to users' preference)

        Model Assumption: None

        Parameters
        ----------
        m: int (>0), default as 10, the dimension of feature/theta
        sigma: float (0,1), default as 0.01, the standard deviation of normal initialization
        eta: float (>0), default as 0.01, learning rate
        lmbda: float (>0), default as 0, L2 regularization parameter
        epoch: int (>0), default as 100, epochs
        N: int (>0), default as 1, number of filters for voting
        d: int (>0), default as 2, the number of decimals presented in prediction matrix

        Returns
        -------
        pred: ndarray, prediction matrix with the same format of input data
        dif: ndarray, difference of input data and prediction matrix, for evaluation
        eval: float, mean absolute error for evaluation
        """
        DAT = self.data.transpose() - self.mu # mean normalization
        DAT = DAT.transpose()
        pred = []

        for n in range(N):

            # parameter initialization
            theta = np.random.rand(m, DAT.shape[1]) * sigma
            feature = np.random.rand(DAT.shape[0], m) * sigma

            # temporary storage container
            theta0 = np.zeros((m, DAT.shape[1]))
            feature0 = np.zeros((DAT.shape[0], m))

            for e in range(epoch):
                # gradient descent for users
                for j in range(DAT.shape[1]):
                    f = self.flag.loc[:, j] == False
                    sub = list(self.flag.loc[f, j].index)
                    for k in range(m):
                        a = np.array([np.dot(theta[:, j], feature[i, :]) for i in sub])
                        b = np.array([DAT[i, j] for i in sub])
                        c = np.array([feature[i, k] for i in sub])
                        delta = eta * (np.dot(a - b, c) + lmbda * theta[k, j])
                        theta0[k, j] = theta[k, j] - delta

                # gradient descent for items
                for i in range(DAT.shape[0]):
                    f = self.flag.loc[i, :] == False
                    sub = list(self.flag.loc[i, f].index)
                    for k in range(m):
                        a = np.array([np.dot(theta[:, j], feature[i, :]) for j in sub])
                        b = np.array([DAT[i, j] for j in sub])
                        c = np.array([theta[k, j] for j in sub])
                        delta = eta * (np.dot(a - b, c) + lmbda * feature[i, k])
                        feature0[i, k] = feature[i, k] - delta

                theta, feature = theta0, feature0

            pred_C = np.dot(feature, theta)
            pred_C = pred_C.transpose() + self.mu  # inverse transformation of mean normalization
            pred_C = pred_C.transpose()
            pred.append(pred_C)

        if N>1:
            pred = np.round(np.sum(np.array(pred), axis=0) / N, decimals=d)  # voting
        elif N==1:
            pred = np.round(np.array(pred), decimals=d).reshape((self.data.shape[0],self.data.shape[1]))
        dif = np.round(self.data - pred, decimals=d)  # difference for evaluation
        ifnan = np.isnan(dif.flatten())
        eval = np.nansum(np.abs(dif)) / (len(ifnan) - sum(ifnan)) # average error for evaluation
        return pred, dif, eval

    def Recommend(self, pred, RatingThreshold):
        """
        Dictionary of Recommendation

        Parameters
        ----------
        pred: ndarray, the first return of DAT.CF method, the prediction matrix
        RatingThreshold: float, a threshold of ratings above which you would
             recommend the corresponding items to users

        Returns
        -------
        catalog: dict, the dictionary of recommendations
        """
        catalog = {}
        for user in range(self.data.shape[1]):
            df = pd.DataFrame(pred)
            log = np.array(self.flag.loc[:, user] == True)*np.array(df.loc[:, user] >= RatingThreshold)
            items = list(self.flag.loc[log, user].index)
            catalog[user] = items
        return catalog
