""" Code from https://github.com/julianmack/Data_Assimilation with only minor modifications """

"""All VarDA ingesting and evaluation helpers"""

import numpy as np
import os
import random
import torch
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

from UnstructuredCAEDA import ML_utils
from UnstructuredCAEDA.AEs import Jacobian
from UnstructuredCAEDA import fluidity
from UnstructuredCAEDA.data.split import SplitData
from UnstructuredCAEDA.VarDA import VDAInit
from UnstructuredCAEDA.VarDA import SVD
from UnstructuredCAEDA.VarDA.cost_fn import cost_fn_J, grad_J

from UnstructuredCAEDA.UnstructuredMesh.HelpersUnstructuredMesh import *

import time

class DAPipeline():
    """Class to hold pipeline functions for Variational DA
    """

    def __init__(self, settings, AEmodel=None, u_c=None):
        self.settings = settings
        vda_initilizer = VDAInit(self.settings, AEmodel, u_c=u_c, case_3d= settings.THREE_DIM, case_2d=settings.TWO_DIM, case_1d=settings.ONE_DIM)
        self.data = vda_initilizer.run() #returns dictionary data

    def run(self, return_stats=False):
        """Runs the variational DA routine using settings from the passed config class
        (see config.py for example)"""
        settings = self.settings

        if settings.COMPRESSION_METHOD == "SVD":
            DA_results = self.DA_SVD()
        elif settings.COMPRESSION_METHOD == "AE":
            DA_results = self.DA_AE()
        else:
            raise ValueError("COMPRESSION_METHOD must be in {SVD, AE}")
        w_opt = DA_results["w_opt"]
        self.print_DA_results(DA_results)

        if return_stats:
            assert return_stats == True, "return_stats must be of type boolean. Here it is type {}".format(type(return_stats))
            stats = {}
            stats["Percent_improvement"] = 100*(DA_results["ref_MAE_mean"] - DA_results["da_MAE_mean"])/DA_results["ref_MAE_mean"]
            stats["ref_MAE_mean"] = DA_results["ref_MAE_mean"]
            stats["da_MAE_mean"] = DA_results["da_MAE_mean"]
            stats["mse_ref"] =  DA_results["mse_ref"]
            stats["mse_DA"] = DA_results["mse_DA"]
            return w_opt, stats

        return w_opt

    def DA_AE(self, trainSteps, testNb, force_init=False, save_vtu=False):
        if self.data.get("model") == None or force_init:
            self.model = ML_utils.load_model_from_settings(self.settings, self.data.get("device"))
            self.data["model"] = self.model
        else:
            self.model = self.data.get("model")

        self.data["model"].eval()

        if self.settings.REDUCED_SPACE:
            if self.data.get("V_trunc") is None or force_init:
                V_red = VDAInit.create_V_red(self.data.get("train_X"),
                                            self.data.get("encoder"),
                                            self.settings)
                self.data["V_trunc"] = V_red.T #tau x M

                self.data["w_0"] = np.zeros((V_red.shape[0]))
                if self.data["G"].diagonal().all() == 1:
                    self.data["G_V"] =self.data["V_trunc"]
                else:
                    self.data["G_V"] = (self.data["G"] @ self.data["V_trunc"] ).astype(float)

            self.data["V_grad"] = None
        else:
            # Now access explicit gradient function
            self.data["V_grad"] = self.__maybe_get_jacobian()

        DA_results = self.perform_VarDA(self.data, self.settings, trainSteps, testNb, save_vtu=save_vtu)
        return DA_results

    def DA_SVD(self, trainSteps, testNb, force_init=False, save_vtu=False):
        if self.data.get("V") is None or force_init:
            V = VDAInit.create_V_from_X(self.data.get("train_X"), self.settings)

            if self.settings.THREE_DIM:
                #(M x nx x ny x nz)
                V = V.reshape((V.shape[0], -1)).T #(n x M)
            else:
                #(M x n)
                V = V.T #(n x M)
            V_trunc, U, s, W = SVD.TSVD(V, self.settings, self.settings.get_number_modes())

            #Define intial w_0
            V_trunc_plus = SVD.SVD_V_trunc_plus(U, s, W, self.settings.get_number_modes())
            
            if self.settings.NORMALIZE:
                w_0 = V_trunc_plus @ np.zeros_like(self.data["u_0"].flatten()) #i.e. this is the value given in Rossella et al (2019).
            else:
                w_0 = V_trunc_plus @ self.data["u_0"].flatten()
            #w_0 = np.zeros((W.shape[-1],)) #TODO - I'm not sure about this - can we assume is it 0?

            self.data["V_trunc"] = V_trunc
            self.data["V"] = V
            self.data["w_0"] = w_0
            self.data["V_grad"] = None

            if self.data.get("G") == 1:
                self.data["G_V"] = self.data["V_trunc"]
            elif self.data.get("G") is None:
                assert self.data.get("obs_idx") is not None
                self.data["G_V"] = self.data["V_trunc"][self.data.get("obs_idx")]
            else:
                raise ValueError("G has be deprecated in favour of `obs_idx`. It should be None")
        DA_results = self.perform_VarDA(self.data, self.settings, trainSteps, testNb, save_vtu=save_vtu)
        return DA_results

    @staticmethod
    def perform_VarDA(data, settings, trainSteps, testNb, save_vtu=False, save_every=60):
        """This is a static method so that it can be performed in AE_train with user specified data"""
        timing_debug = True

        args = (data, settings)
        w_0 = data.get("w_0")
        if w_0 is None:
            raise ValueError("w_0 was not initialized")
        
        t1 = time.time() #before minimisation for each test case
        res = minimize(cost_fn_J, data.get("w_0"), args = args, method='L-BFGS-B',
                jac=grad_J, tol=settings.TOL)
        t2 = time.time()
        string_out = "minCostFunction = {:.4f} s, ".format(t2 - t1)

        w_opt = res.x
        u_0 = data.get("u_0")
        u_c = data.get("u_c")
        std = data.get("std")
        mean = data.get("mean")

        if settings.DEBUG:

            size = len(u_0.flatten())
            if size > 10:
                size = 10
            print("Following minimisation and before DA stats:")
            print("std:    ", std.flatten()[:size])
            print("mean:   ", mean.flatten()[:size])
            print("u_0:    ", u_0.flatten()[:size])
            print("u_c:    ", u_c.flatten()[:size])
            print("W_opt: ", w_opt.flatten()[:size])


        if settings.COMPRESSION_METHOD == "SVD":
            delta_u_DA = (data.get("V_trunc") @ w_opt).flatten()
            u_0 = u_0.flatten()
            u_c = u_c.flatten()
            std = std.flatten()
            mean = mean.flatten()
            u_DA = u_0 + delta_u_DA
            out_str_2 = ""

        elif settings.COMPRESSION_METHOD == "AE" and settings.REDUCED_SPACE:
            # Found optimal w in reduced space to need to
            # V * w then convert to full space
            # using V_trunc as is V_reduced.T
            ta = time.time()
            q_opt = data.get("V_trunc") @ w_opt
            tb = time.time()
            x1, x3, x6, x8, x10 = data.get("decoderOutputDimPerLayer")

            delta_u_DA  = data.get("decoder")(q_opt, x1, x3, x6, x8, x10, encoder=False)
            tc = time.time()
            u_DA = u_0 + delta_u_DA
            td = time.time()
            out_str_AE_steps = "v_trunc (Latent to Reduced) = {:.4f}, dec (Reduced to Full) = {:.4f}, add (DA)= {:.4f}"
            out_str_AE_steps = out_str_AE_steps.format(tb - ta, tc - tb, td-tc)
            string_out += out_str_AE_steps

        elif settings.COMPRESSION_METHOD == "AE":
            delta_u_DA = data.get("decoder")(w_opt)
            if settings.THREE_DIM and len(delta_u_DA.shape) != 3:
                delta_u_DA = delta_u_DA.squeeze(0)

            u_DA = u_0 + delta_u_DA

        t3 = time.time()
        t4 = time.time()
        string_out += "decode = {:.4f} s, ".format(t3 - t2)
        if settings.UNDO_NORMALIZE:
            u_DA = (u_DA * std + mean)
            t4 = time.time() 
            u_c = (u_c * std + mean)
            u_0 = (u_0 * std + mean)
        elif settings.NORMALIZE:
            t4 = time.time()

        if settings.DEBUG:
            print("Following Assimilation:")
            print("std:    ", std.flatten()[:size])
            print("mean:   ", mean.flatten()[:size])
            print("u_0:    ", u_0.flatten()[:size])
            print("u_c:    ", u_c.flatten()[:size])
            print("u_DA: ", u_DA.flatten()[:size])

            string_out += "unnorm = {:.4f} s, ".format(t4 - t3)
            string_out += "TOTAL = unnormalising + decoding + minimising = {:.4f} s, ".format(t4 - t1)

        t_statStart = time.time()
        
        ref_MAE = np.abs(u_0 - u_c)
        da_MAE = np.abs(u_DA - u_c)
        ref_MAE_mean = np.mean(ref_MAE)
        da_MAE_mean = np.mean(da_MAE)
        
        percent_improvement = 100 * (ref_MAE_mean - da_MAE_mean)/ref_MAE_mean
        counts = (da_MAE < ref_MAE).sum()
        mse_ref = np.linalg.norm(u_0 - u_c) /  np.linalg.norm(u_c)
        mse_DA = np.linalg.norm(u_DA - u_c) /  np.linalg.norm(u_c)

        if settings.TWO_DIM:
            mse_ref_points = mse_ref / (np.shape(u_c)[0] * np.shape(u_c)[1])
            mse_da_points = mse_DA / (np.shape(u_c)[0] * np.shape(u_c)[1])

        elif settings.ONE_DIM:
            mse_ref_points = mse_ref / len(u_c)
            mse_da_points = mse_DA / len(u_c)
            
        # Big Data Overlapping regions evaluation statistics
        t_overlap_start = time.time()
        idxOverlappingRegions = settings.OVERLAPPING_REGIONS
        idxMeshToIdxNetwork = settings.IDX_MESH_TO_IDX_NETWORK_MATCH
        u_c_overlap = DAPipeline.getOverlappingRegions(u_c, idxMeshToIdxNetwork)
        u_0_overlap = DAPipeline.getOverlappingRegions(u_0, idxMeshToIdxNetwork)
        u_da_overlap = DAPipeline.getOverlappingRegions(u_DA, idxMeshToIdxNetwork)
        t_overlap_end = time.time()
        mse_ref_overlap = DAPipeline.getMSE(u_0_overlap, u_c_overlap)
        mse_da_overlap = DAPipeline.getMSE(u_da_overlap, u_c_overlap)

        t_save_uc_start = time.time()
        t_save_uc_end = time.time()
        t_save_da_end = time.time()
        t_save_u0_end = time.time()
        
        # Visualise on vtu grid the DA process
        if testNb % save_every == 0:
            t_save_uc_start = time.time()
            saveGrid(u_c, settings, False, "controlState", trainSteps=trainSteps, idx=testNb)
            t_save_uc_end = time.time()
            saveGrid(u_DA, settings, False, "da", trainSteps=trainSteps, idx=testNb)
            t_save_da_end = time.time()
            saveGrid(u_0, settings, False, "background", trainSteps=trainSteps, idx=testNb)
            t_save_u0_end = time.time()

        mse_percent_overlap = 100 * (mse_ref_overlap - mse_da_overlap) / mse_ref_overlap
        mse_percent_points = 100 * (mse_ref_points - mse_da_points) / mse_ref_points

        mse_percent = 100 * (mse_ref - mse_DA)/mse_ref     

        results_data = {"u_DA": u_DA,
                    "ref_MAE_mean": ref_MAE_mean,
                    "da_MAE_mean": da_MAE_mean,
                    "percent_improvement": percent_improvement,
                    "counts": counts,
                    "w_opt": w_opt,
                    "mse_ref": mse_ref,
                    "mse_DA": mse_DA,
                    "mse_percent": mse_percent,
                    "mse_percent_points": mse_percent_points,
                    "mse_percent_overlap": mse_percent_overlap,
                    "time_online": t4 - t1,
                    "time_saving_uc": t_save_uc_end - t_save_uc_start,
                    "time_saving_da": t_save_da_end - t_save_uc_end,
                    "time_saving_u0": t_save_u0_end - t_save_da_end,
                    "time_overlap": t_overlap_start - t_overlap_end,
                    "mse_da_points": mse_da_points,
                    "mse_ref_points": mse_ref_points,
                    "mse_ref_overlap": mse_ref_overlap,
                    "mse_da_overlap": mse_da_overlap}
        if save_vtu:
            results_data["ref_MAE"] = ref_MAE.flatten()
            results_data["da_MAE"]  = da_MAE.flatten()

        if settings.DEBUG:
            t5 = time.time()
            string_out += "inc stats = {:.4f}, ".format(t5- t1)
            print(string_out)

        if settings.SAVE:
            if False:
                out_fp_ref = settings.INTERMEDIATE_FP + "ref_MAE.vtu"
                out_fp_DA =  settings.INTERMEDIATE_FP + "da_MAE.vtu"
                fluidity.utils.save_vtu(settings, out_fp_ref, ref_MAE)
                fluidity.utils.save_vtu(settings, out_fp_DA, da_MAE)

        if settings.DEBUG:

            size = len(u_0.flatten())
            if size > 5:
                size = 5
            print("std:    ", std.flatten()[:size])
            print("mean:   ", mean.flatten()[:size])
            print("u_0:    ", u_0.flatten()[:size])
            print("u_c:    ", u_c.flatten()[:size])
            print("u_DA:   ", u_DA.flatten()[:size])
            print("ref_MAE:", ref_MAE.flatten()[:size])
            print("da_MAE: ", da_MAE.flatten()[:size])
            print("%", percent_improvement, "da_MAE", da_MAE_mean,"ref_MAE", ref_MAE_mean)

        return results_data

    @staticmethod
    def getMSE(u_1, u_2):
        u_1 = np.array(u_1)
        u_2 = np.array(u_2)
        mse = np.square(u_1 - u_2).mean()
        numerator = np.linalg.norm(u_1 - u_2)
        normalising = np.linalg.norm(u_2)
        return numerator / normalising

    @staticmethod
    def getOverlappingRegions(u, idxMeshToIdxNetwork):
        u_overlap = []
        u_flat = u.flatten()
        networkIndices = []
        idxOverlappingRegions = idxMeshToIdxNetwork.keys()
        for idx in idxOverlappingRegions:
            networkIdx = idxMeshToIdxNetwork.get(idx)
            networkIndices.append(networkIdx)
        for element in networkIndices:
            if element < len(u_flat): #because when reshaping a few locations are dismissed (up to 90)
                u_overlap.append(u_flat[element])
        return u_overlap

    def __maybe_get_jacobian(self):
        jac = None
        if not self.settings.JAC_NOT_IMPLEM:
            try:
                jac = self.model.jac_explicit
            except:
                pass
        else:
            import warnings
            warnings.warn("Using **Very** slow method of calculating jacobian. Consider disabling DA", UserWarning)
            jac = self.slow_jac_wrapper

        if jac == None:
            raise NotImplementedError("This model type does not have a gradient available")
        return jac

    def slow_jac_wrapper(self, x):
        return Jacobian.accumulated_slow_model(x, self.model, self.data.get("device"))

    @staticmethod
    def print_DA_results(DA_results):

        u_DA = DA_results["u_DA"]
        ref_MAE_mean = DA_results["ref_MAE_mean"]
        da_MAE_mean = DA_results["da_MAE_mean"]
        w_opt = DA_results["w_opt"]
        counts = DA_results["counts"]
        mse_ref = DA_results["mse_ref"]
        mse_DA = DA_results["mse_DA"]
        print("Ref MAE: {:.4f}, DA MAE: {:.4f},".format(ref_MAE_mean, da_MAE_mean), "% improvement: {:.2f}%".format(DA_results["percent_improvement"]))
        print("DA_MAE < ref_MAE for {}/{} points".format(counts, len(u_DA.flatten())))
        print("MSE _percent {:.4f}, ")
        print("mse_ref: {:.4f}, mse_DA: {:.4f}".format(mse_ref, mse_DA))
        #Compare abs(u_0 - u_c).sum() with abs(u_DA - u_c).sum() in paraview

if __name__ == "__main__":

    settings = config.Config()

    DA = DAPipeline(settings)
    DA.run()
