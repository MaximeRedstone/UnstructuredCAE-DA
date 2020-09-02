""" Code from https://github.com/julianmack/Data_Assimilation with only minor modifications """

import numpy as np
import random

from UnstructuredCAEDA import ML_utils

class SplitData:

    @staticmethod
    def train_test_DA_split_maybe_normalize(X, settings):
        """Returns non-overlapping train/test and DA control state data.
        This function also deals with normalization (to ensure than only the
        training data is used for normalization mean and std)"""

        M, n = SplitData.get_dim_X(X, settings)
        
        hist_idx = int(M * settings.HIST_FRAC) #using 4/5
        hist_X = X[: hist_idx] #select historical data (i.e. training set in ML terminology)
                                 # that will be used for normalize

        #use only the training set to calculate mean and std
        mean = np.mean(hist_X, axis=0)
        std = np.std(hist_X, axis=0)

        #Some std are zero - set the norm to 1 in this case so that feature is zero post-normalization
        std = np.where(std <= 0., 1, std)

        # print("x before normalisation: ", X)
        print("Splitting and setting training testing set: Hist frac = {}, hist IDX = {}".format(settings.HIST_FRAC, hist_idx))
        
        if settings.NORMALIZE:
            X = (X - mean)
            X = (X / std)
        else:
            X = np.asarray(X)

        # Split X into historical and present data. We will
        # assimilate "observations" at a single timestep t_DA
        # which corresponds to the control state u_c
        # We will take initial condition u_0, as mean of historical data

        t_DA = M - (settings.TDA_IDX_FROM_END + 1) #idx of Data Assimilation
        assert t_DA >= hist_idx, ("Cannot select observation from historical data."
                                "Reduce HIST_FRAC or reduce TDA_IDX_FROM_END to prevent overlap.\n"
                                "t_DA = {} and hist_idx = {}".format(t_DA, hist_idx))
        assert t_DA > hist_idx, ("Test set cannot have zero size")

        train_X = X[: hist_idx]
        test_X = X[hist_idx : t_DA]
        u_c = X[t_DA] #control state (for DA)
        print("Train from 0 to {} - test from {} to {} and t_DA at {}".format(hist_idx, hist_idx, t_DA, t_DA))

        if settings.SHUFFLE_DATA:
            ML_utils.set_seeds()
            np.random.shuffle(train_X)
            np.random.shuffle(test_X)

        return train_X, test_X, u_c, X, mean, std

    @staticmethod
    def get_dim_X(X, settings):

        if settings.THREE_DIM:
            M, nx, ny, nz = X.shape
            n = (nx, ny, nz)
        elif settings.TWO_DIM:
            M, nx, ny = X.shape
            n = (nx, ny)
        else:
            M, n = np.shape(X)[0], np.shape(X)[1]
        # assert n == settings.get_n(), "dimensions {} must = {}".format(n, settings.get_n())
        return M, n