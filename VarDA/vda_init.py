""" Code from https://github.com/julianmack/Data_Assimilation with modifications:
- to fit 1D assimilation 
- conduct DA using a single early time step """

import numpy as np
import random
import torch

from UnstructuredCAEDA import ML_utils
from UnstructuredCAEDA.data.split import SplitData

class VDAInit:
    def __init__(self, settings, AEmodel=None, u_c=None, case_3d=True, case_2d=False, case_1d=False):
        self.AEmodel = AEmodel
        if self.AEmodel is not None:
            self.AEmodel.eval()
        self.settings = settings
        self.u_c = u_c #user can pass their own u_c
        self.CASE_3D = case_3d
        self.CASE_1D = case_1d
        self.CASE_2D = case_2d

    def run(self):
        """Generates matrices for VarDA. All returned matrices are in the
        (M X n) or (M x nx x ny x nz) format """

        print("-----------------------------DA Init----------------------")
        data = {}
        loader = self.settings.get_loader()
        splitter = SplitData()
        settings = self.settings

        X = loader.get_X(settings)

        train_X, test_X, u_c_std, X, mean, std = splitter.train_test_DA_split_maybe_normalize(X, settings)
        
        if self.u_c is None:
            self.u_c = u_c_std

        #self.u_c = train_X[62] #good
        #self.u_c = train_X[-1] #bad

        # We will take initial condition u_0, as mean of historical data or as a single timestep
        if int(self.settings.MEAN_HIST_DATA):
            u_0 = np.zeros_like(mean) #since the data is mean centred
        else:
            u_0 = train_X[settings.DA_BACKGROUND_IDX]

        encoder = None
        decoder = None
        
        device = ML_utils.get_device()
        model = self.AEmodel
        if model:
            model.to(device)

        if self.settings.COMPRESSION_METHOD == "AE":
            if model is None:
                model = ML_utils.load_model_from_settings(settings)
                print("Model loaded", sum(p.numel() for p in self.model.parameters()))

            def __create_encoderOrDecoder3D(fn, x1=None, x3=None, x6=None, x8=None, x10=None, encoder=True):
                """This returns a function that deals with encoder/decoder
                input dimensions (e.g. adds channel dim for 3D case)"""
                def ret_fn(vec, x1=None, x3=None, x6=None, x8=None, x10=None, encoder=True):
                    vec = torch.Tensor(vec).to(device)

                    #for 3D case, unsqueeze for channel
                    if self.settings.THREE_DIM:
                        dims = len(vec.shape)
                        if dims == 3:
                            vec = vec.unsqueeze(0)
                        elif dims == 4:
                            #batched input
                            vec = vec.unsqueeze(1)

                    with torch.no_grad():
                        if encoder:
                            res, x1, x3, x6, x8, x10 = fn(vec)
                            res = res.detach().cpu()
                        else:
                            res = fn(vec, x1, x3, x6, x8, x10)

                    #for 3D case, squeeze for channel
                    dims = len(res.shape)
                    if self.settings.THREE_DIM and dims > 2:
                        if dims == 4:
                            res = res.squeeze(0)
                        elif dims == 5:   #batched input
                            res = res.squeeze(1)
                    
                    if encoder:
                        return res.numpy(), x1, x3, x6, x8, x10
                    else:
                        return res.numpy()

                if encoder:
                    return ret_fn, x1, x3, x6, x8, x10
                else:
                    return ret_fn

            def __create_encoderOrDecoder2D(fn, x1=None, x3=None, x6=None, x8=None, x10=None, encoder=True):
                """This returns a function that deals with encoder/decoder
                input dimensions (e.g. adds channel dim for 3D case)"""
                def ret_fn(vec, x1=None, x3=None, x6=None, x8=None, x10=None, encoder=True):
                    vec = torch.Tensor(vec).to(device)

                    #for 2D case, unsqueeze for channel
                    if self.settings.TWO_DIM:
                        dims = len(vec.shape)
                        if dims == 2:
                            vec = vec.unsqueeze(0)
                        elif dims == 3:
                            #batched input
                            vec = vec.unsqueeze(1)

                    with torch.no_grad():
                        if encoder:
                            res, x1, x3, x6, x8, x10 = fn(vec)
                            res = res.detach().cpu()
                        else:
                            res = fn(vec, x1, x3, x6, x8, x10)

                    #for 3D case, squeeze for channel
                    dims = len(res.shape)
                    if self.settings.TWO_DIM:
                        if dims == 3:
                            res = res.squeeze(0)
                        elif dims == 4:   #batched input
                            res = res.squeeze(1)
                    
                    if encoder:
                        return res.numpy(), x1, x3, x6, x8, x10
                    else:
                        return res.numpy()

                if encoder:
                    return ret_fn, x1, x3, x6, x8, x10
                else:
                    return ret_fn

            def __create_encoderOrDecoder1D(fn, x1=None, x3=None, x6=None, x8=None, x10=None, encoder=True):
                """This returns a function that deals with encoder/decoder
                input dimensions (e.g. adds channel dim for 3D case)"""
                def ret_fn(vec, x1=None, x3=None, x6=None, x8=None, x10=None, encoder=True):
                    vec = torch.Tensor(vec).to(device)

                    #for 1D case, unsqueeze for channel
                    if self.settings.ONE_DIM:
                        dims = len(vec.shape)
                        if dims == 1: #2
                            vec = vec.unsqueeze(0)
                        elif dims == 2: #3
                            #batched input
                            vec = vec.unsqueeze(1)

                    with torch.no_grad():
                        if encoder:
                            res, x1, x3, x6, x8, x10 = fn(vec)
                            res = res.detach().cpu()
                        else:
                            res = fn(vec, x1, x3, x6, x8, x10)
            
                    #for 1D case, squeeze for channel
                    dims = len(res.shape)
                    if self.settings.ONE_DIM:
                        if dims == 2:
                            res = res.squeeze(0)
                        elif dims == 3:   #batched input
                            res = res.squeeze(1)
                    
                    # print("Returning res shape = {} and value = {}".format(np.shape(res), res))
                    if encoder:
                        return res.numpy(), x1, x3, x6, x8, x10
                    else:
                        return res.numpy()

                if encoder:
                    return ret_fn, x1, x3, x6, x8, x10
                else:
                    return ret_fn
            
            if self.CASE_3D:
                encoder, x1, x3, x6, x8, x10 = __create_encoderOrDecoder3D(model.encode, encoder=True)
                decoder = __create_encoderOrDecoder3D(model.decode, x1, x3, x6, x8, x10, encoder=False)
            elif self.CASE_2D:
                encoder, x1, x3, x6, x8, x10 = __create_encoderOrDecoder2D(model.encode, encoder=True)
                decoder = __create_encoderOrDecoder2D(model.decode, x1, x3, x6, x8, x10, encoder=False)
            elif self.CASE_1D:
                encoder, x1, x3, x6, x8, x10 = __create_encoderOrDecoder1D(model.encode, encoder=True)
                decoder = __create_encoderOrDecoder1D(model.decode, x1, x3, x6, x8, x10, encoder=False)

        H_0, obs_idx = None, None

        if self.settings.REDUCED_SPACE == True:
            if self.settings.COMPRESSION_METHOD == "SVD":
                raise NotImplementedError("SVD in reduced space not implemented")

            self.settings.OBS_MODE = "all"

            observations, H_0, w_0, d, decoderOuputDimPerLayer = self.__get_obs_and_d_reduced_space(self.settings, self.u_c, u_0, encoder)
            
            data = {"d": d, "G": H_0,
                    "observations": observations,
                    "model": model, "obs_idx": obs_idx,
                    "encoder": encoder, "decoder": decoder,
                    "u_c": self.u_c, "u_0": u_0, "X": X,
                    "train_X": train_X, "test_X":test_X,
                    "std": std, "mean": mean, "device": device,
                    "decoderOutputDim": decoderOuputDimPerLayer}
        else:
            observations, w_0, d, obs_idx = self.__get_obs_and_d_not_reduced(self.settings, self.u_c, u_0, encoder)

        data = {"d": d, "G": H_0,
                "observations": observations,
                "model": model, "obs_idx": obs_idx,
                "encoder": encoder, "decoder": decoder,
                "u_c": self.u_c, "u_0": u_0, "X": X,
                "train_X": train_X, "test_X":test_X,
                "std": std, "mean": mean, "device": device}
   
        if w_0 is not None:
            data["w_0"] = w_0

        return data

    @staticmethod
    def create_V_from_X(X_fp, settings):
        """Creates a mean centred matrix V from input matrix X.
        X_FP can be a numpy matrix or a fp to X.
        returns V in the  M x n format"""

        if type(X_fp) == str:
            X = np.load(X_fp)
        elif type(X_fp) == np.ndarray:
            X = X_fp
        else:
            raise TypeError("X_fp must be a filpath or a numpy.ndarray")

        M, n = SplitData.get_dim_X(X, settings)

        mean = np.mean(X, axis=0)

        if settings.NORMALIZE:
            assert np.allclose(mean, np.zeros_like(mean))

        V = (X - mean)

        return V

    @staticmethod
    def select_obs(settings, vec):
        """Selects and return a subset of observations and their indexes
        from vec according to a user selected mode"""
        npoints = VDAInit.__get_npoints_from_shape(vec.shape)
        if settings.OBS_MODE == "rand":
            # Define observations as a random subset of the control state.
            if hasattr(settings, "NOBS"):
                nobs = settings.NOBS
            else:
                nobs = int(settings.OBS_FRAC * npoints) #number of observations
            assert nobs <= npoints, "You can't select more observations that are in the state space"
            if nobs == npoints: #then we are selecting all points
                settings.OBS_MODE = "all"
                return VDAInit.__select_all_obs(vec)
            ML_utils.set_seeds(seed = settings.SEED) #set seeds so that the selected subset is the same every time
            obs_idx = random.sample(range(npoints), nobs) #select nobs integers w/o replacement
            observations = np.take(vec, obs_idx)
        elif settings.OBS_MODE == "single_max":
            nobs = 1
            obs_idx = np.argmax(vec)
            obs_idx = [obs_idx]
            observations = np.take(vec, obs_idx)
        elif settings.OBS_MODE == "all":
            observations, obs_idx, nobs = VDAInit.__select_all_obs(vec)
        else:
            raise ValueError("OBS_MODE = {} is not allowed.".format(settings.OBS_MODE))
        assert nobs == len(obs_idx)
        return observations, obs_idx, nobs

    @staticmethod
    def create_H(obs_idxs, n, nobs, three_dim=False, obs_mode=None):
        """Creates the mapping matrix from the statespace to the observations.
            :obs_idxs - an iterable of indexes @ which the observations are made
            :n - size of state space
            :nobs - number of observations
        returns
            :H - numpy array of size (nobs x n)
        """
        if three_dim:
             nx, ny, nz = n
             n = nx * ny * nz
        else:
            assert type(n) == int
        if obs_mode == "all":
            return 1 #i.e. identity but no need to store full size
        H = np.zeros((nobs, n))
        H[range(nobs), obs_idxs] = 1

        assert H[0, obs_idxs[0]] == 1, "1s were not correctly assigned"
        assert H[0, (obs_idxs[0] + 1) % n ] == 0, "0s were not correctly assigned"
        assert H[nobs - 1, obs_idxs[-1]] == 1, "1s were not correctly assigned"
        assert H.shape == (nobs, n)

        return H

    @staticmethod
    def create_R_inv(sigma, nobs):
        """Creates inverse of R: the observation error matrix.
        Assume all observations are independent s.t. R = sigma**2 * identity.
        args
            :sigma - observation error variance
            :nobs - number of observations
        returns
            :R_inv - (nobs x nobs) array"""

        R_inv = 1.0 / sigma ** 2 * np.eye(nobs)

        assert R_inv.shape == (nobs, nobs)

        return R_inv

    @staticmethod
    def __get_obs_and_d_not_reduced(settings, u_c, u_0, encoder=None):
        if settings.COMPRESSION_METHOD == "AE":
            if encoder is None:
                raise ValueError("Encoder must be provided if settings.COMPRESSION_METHOD == `AE` ")
            w_0 = encoder(u_0)
            #w_0_v1 = torch.zeros((settings.get_number_modes())).to(device)
        else:
            w_0 = None #this will be initialized in SVD_DA()
        observations, obs_idx, nobs = VDAInit.select_obs(settings, u_c) #options are specific for rand

        d = observations.flatten() - u_0.flatten()[obs_idx]
        assert settings.COMPRESSION_METHOD == "SVD"

        return observations, w_0, d, obs_idx

    @staticmethod
    def __get_obs_and_d_reduced_space(settings, u_c, u_0, encoder):
        """helper function to get observations and H_0 and d in reduced space"""
        if len(u_c.shape) not in [1, 3] and settings.THREE_DIM:
            raise ValueError("This function does not accept batched input with {} dimensions".format(len(u_c.shape)))
        elif len(u_c.shape) != 2 and settings.TWO_DIM:
            raise ValueError("This function does not accept batched input with {} dimensions for 2D case".format(len(u_c.shape)))
        
        z_c, x1, x3, x6, x8, x10 = encoder(u_c)
        H_0 = np.eye(len(z_c))

        if int(settings.MEAN_HIST_DATA):
            d = z_c
        elif settings.NORMALIZE:
            z_0, x1, x3, x6, x8, x10 = encoder(u_0)
            d = z_c - z_0

        observations = z_c
        decoderOuputDimPerLayer = x1, x3, x6, x8, x10
        return observations, H_0, None, d, decoderOuputDimPerLayer #None is w_0

    @staticmethod
    def provide_u_c_update_data_full_space(data, settings, u_c):
        assert settings.REDUCED_SPACE == False, "This function only works in reduced space"
        encoder = data.get("encoder")
        u_0 = data.get("u_0")
        if encoder is None and settings.COMPRESSION_METHOD == "AE":
            raise ValueError("Encoder must be initialized in `data` dict")
        if u_0 is None:
            raise ValueError("u_0 must be initialized in `data` dict")

        observations, w_0, d, obs_idx= VDAInit.__get_obs_and_d_not_reduced(settings, u_c, u_0, encoder)
        data["observations"] = observations
        data["obs_idx"] = obs_idx
        if w_0 is not None: #i.e. don't update if no result was returned
            data["w_0"] = w_0

        data["d"] = d
        data["u_c"] = u_c
        return data

    @staticmethod
    def provide_u_c_update_data_reduced_AE(data, settings, u_c):
        assert settings.REDUCED_SPACE == True, "This function only works in reduced space"
        assert settings.COMPRESSION_METHOD == "AE", "This function only works for AE"
        encoder = data.get("encoder")
        u_0 = data.get("u_0")
        if encoder is None:
            raise ValueError("Encoder must be initialized in `data` dict")
        if u_0 is None:
            raise ValueError("u_0 must be initialized in `data` dict")

        # print("Running provide uc update data reduced space")
        observations, H_0, w_0, d, decoderOuputDimPerLayer = VDAInit.__get_obs_and_d_reduced_space(settings, u_c, u_0, encoder)
        data["observations"] = observations
        data["G"] = H_0
        data["d"] = d
        data["u_c"] = u_c
        data["decoderOutputDimPerLayer"] = decoderOuputDimPerLayer

        if w_0 is not None: #i.e. don't update if no result was returned
            data["w_0"] = w_0
        return data

    @staticmethod
    def create_V_red(X, encoder, settings, number_modes=None):
        V = VDAInit.create_V_from_X(X, settings)
        res = []
        BATCH = V.shape[0] if V.shape[0] <= 16 else 16
        assert BATCH <= 16, "Batch must be <=16 and BATCH = {}".format(BATCH)

        i_start = 0
        for i in range(BATCH, V.shape[0] + BATCH, BATCH):

            idx_end = i if i < V.shape[0] else V.shape[0]
            inp = V[i_start:idx_end]
            
            v, x1, x3, x6, x8, x10 = encoder(inp)
            res.append(v)

            i_start = i
            
        V_red = np.concatenate(res, axis=0)

        if number_modes:
            #take evenly spaced states
            step = V_red.shape[0] / number_modes
            idxs = [int(x * step) for x in range(number_modes)]

            assert len(idxs) == number_modes

            V_red = V_red[idxs]

        return V_red

    @staticmethod
    def __select_all_obs(vec):
        nobs = VDAInit.__get_npoints_from_shape(vec.shape)
        return vec, list(range(nobs)), nobs

    @staticmethod
    def __get_npoints_from_shape(n):
        if type(n) == tuple:
            npoints = 1
            for val in n:
                npoints *= val
        elif type(n) == int:
            npoints = n
        else:
            raise TypeError("Size n must be of type int or tuple")
        return npoints
