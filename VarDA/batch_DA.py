""" Code from https://github.com/julianmack/Data_Assimilation with modifications:
- Big Data Assimilation results statistics """

import torch
import time

from UnstructuredCAEDA.data.split import SplitData
from UnstructuredCAEDA import ML_utils, fluidity
from UnstructuredCAEDA.VarDA import DAPipeline
from UnstructuredCAEDA.VarDA import SVD, VDAInit
from UnstructuredCAEDA.utils.expdir import init_expdir
from UnstructuredCAEDA.settings import helpers

from UnstructuredCAEDA.UnstructuredMesh.HelpersUnstructuredMesh import *

import pandas as pd
import numpy as np

class BatchDA():
    def __init__(self, settings, control_states=None, csv_fp=None, csv_avg_fp=None, AEModel=None,
                reconstruction=True, plot=False, save_vtu=False, totalTimesteps=None):

        self.settings = settings
        self.control_states = control_states
        self.reconstruction = reconstruction
        self.plot = plot
        self.model = AEModel
        self.csv_fp = csv_fp
        self.csv_avg_fp = csv_avg_fp
        self.save_vtu = save_vtu
        self.totalTimesteps = totalTimesteps

        print("\n\n----------------------------- START OF DA ----------------------")

        if self.csv_fp and self.csv_avg_fp:
            fps = self.csv_fp.split("/")
            fps_avg = self.csv_avg_fp.split("/")
            dir = fps[:-1]
            dir = "/".join(fps[:-1])
            self.expdir = init_expdir(dir, True)
            self.file_name = fps[-1]
            self.file_name_avg = fps_avg[-1]
            self.csv_fp = helpers.win_to_unix_fp(self.expdir + self.file_name)
            self.csv_avg_fp = helpers.win_to_unix_fp(self.expdir + self.file_name_avg)

        if self.save_vtu:
            if self.csv_fp:
                self.save_vtu_fp = self.csv_fp.replace(".csv", "")
            else:
                raise ValueError("Must past csv fp to save vtu file")

        if self.control_states is None:
            loader, splitter = settings.get_loader(), SplitData()
            X = loader.get_X(settings)

            print("----------------------------- DA Loading Data and Splitting ----------------------")
            train_X, test_X, u_c_std, X, mean, std = splitter.train_test_DA_split_maybe_normalize(X, settings)
            self.control_states = test_X
            self.trainSteps = np.shape(train_X)[0]

    def run(self, print_every=10, print_small=True):

        shuffle = self.settings.SHUFFLE_DATA #save value
        self.settings.SHUFFLE_DATA = False

        if self.settings.COMPRESSION_METHOD == "SVD":
            if self.settings.REDUCED_SPACE:
                raise NotImplementedError("Cannot have reduced space SVD")
            fileDir = "svd_matrices"
            try:
                os.chdir(os.getcwd() + str(self.settings.getExptDir()))
            except OSError:
                pass

            if os.path.exists(fileDir):
                U = np.load(fileDir + "/U.npy")
                s = np.load(fileDir + "/s.npy")
                W = np.load(fileDir + "/W.npy")
    
                num_modes = self.settings.get_number_modes()
                V_trunc = SVD.SVD_V_trunc(U, s, W, modes=num_modes)
                V_trunc_plus = SVD.SVD_V_trunc_plus(U, s, W, modes=num_modes)

                self.DA_pipeline = DAPipeline(self.settings)
                DA_data = self.DA_pipeline.data 
                DA_data["V_trunc"] = V_trunc
                DA_data["V"] = None
                DA_data["w_0"] = V_trunc_plus @ DA_data.get("u_0").flatten()
                DA_data["V_grad"] = None

            else:
                self.DA_pipeline = DAPipeline(self.settings)
                DA_data = self.DA_pipeline.data

            num_modes = self.settings.get_number_modes() #Suppose to return 2 from base.py but now UnstructuredCLIC is base_block not implemented

        elif self.settings.COMPRESSION_METHOD == "AE":
            if self.model is None:
                raise ValueError("Must provide an AE torch.nn model if settings.COMPRESSION_METHOD == 'AE'")

            self.DA_pipeline = DAPipeline(self.settings, self.model)
            DA_data = self.DA_pipeline.data

            if self.reconstruction:
                encoder = DA_data.get("encoder")
                decoder = DA_data.get("decoder")

        else:
            raise ValueError("settings.COMPRESSION_METHOD must be in ['AE', 'SVD']")

        self.settings.SHUFFLE_DATA = shuffle

        if self.reconstruction:
            L1 = torch.nn.L1Loss(reduction='sum')
            L2 = torch.nn.MSELoss(reduction="sum")

        totals = {"percent_improvement": 0,
                "ref_MAE_mean": 0,
                "da_MAE_mean": 0,
                "mse_DA": 0,
                "mse_ref": 0,
                "counts": 0,
                "l1_loss": 0,
                "l2_loss": 0,
                "time": 0,
                "time_online": 0,
                "mse_ref_points": 0,
                "mse_da_points": 0,
                "mse_ref_overlap": 0,
                "mse_da_overlap": 0,
                "mse_percent_points": 0,
                "mse_percent_overlap": 0,
                "t_bigDataStat": 0,
                "time_saving_uc": 0, 
                "time_saving_da": 0,
                "time_saving_u0": 0,
                "time_overlap": 0}

        tot_DA_MAE = np.zeros_like(self.control_states[0]).flatten()
        tot_ref_MAE = np.zeros_like(self.control_states[0]).flatten()
        results = []

        if self.settings.THREE_DIM and len(self.control_states.shape) in [1, 3]:
            raise ValueError("This is not batched control_state input for 3D case")
        elif self.settings.TWO_DIM and len(self.control_states.shape) == 2:
            raise ValueError("This is not batched control_state input for 2D case")
        else:
            num_states = self.control_states.shape[0]
        
        print("----------------------------- DA Testing for each test case ----------------------")
        if int(self.settings.MEAN_HIST_DATA):
            print("Taking mean of historical data = ", self.settings.MEAN_HIST_DATA)

        for idx in range(num_states):
            u_c = self.control_states[idx]
            if self.settings.REDUCED_SPACE:
                self.DA_pipeline.data = VDAInit.provide_u_c_update_data_reduced_AE(DA_data,
                                                                                    self.settings, u_c)
            else:
                self.DA_pipeline.data = VDAInit.provide_u_c_update_data_full_space(DA_data,
                                                                                        self.settings, u_c)
            t1 = time.time()
            if self.settings.COMPRESSION_METHOD == "AE":
                DA_results = self.DA_pipeline.DA_AE(self.trainSteps, idx, save_vtu=self.save_vtu)
            elif self.settings.COMPRESSION_METHOD == "SVD":
                DA_results = self.DA_pipeline.DA_SVD(self.trainSteps, idx, save_vtu=self.save_vtu)
            t2 = time.time()
            t_tot = t2 - t1

            if self.reconstruction:
                data_tensor = torch.Tensor(u_c)
                if self.settings.COMPRESSION_METHOD == "AE":
                    device = ML_utils.get_device()
                    #device = ML_utils.get_device(True, 1)

                    data_tensor = data_tensor.to(device)

                    res, x1, x3, x6, x8, x10 = encoder(u_c, encoder=True)
                    data_hat = decoder(res, x1, x3, x6, x8, x10, encoder=False)
                    data_hat = torch.Tensor(data_hat)
                    data_hat = data_hat.to(device)

                elif self.settings.COMPRESSION_METHOD == "SVD":
                    
                    if os.path.exists(fileDir):
                        U = np.load(fileDir + "/U.npy")
                        s = np.load(fileDir + "/s.npy")
                        W = np.load(fileDir + "/W.npy")

                    data_hat = SVD.SVD_reconstruction_trunc(u_c, U, s, W, num_modes)

                    data_hat = torch.Tensor(data_hat)
                with torch.no_grad():
                    l1 = L1(data_hat, data_tensor)
                    l2 = L2(data_hat, data_tensor)
            else:
                l1, l2 = None, None


            result = {}
            result["percent_improvement"] = DA_results["percent_improvement"]
            result["ref_MAE_mean"] =  DA_results["ref_MAE_mean"]
            result["da_MAE_mean"] = DA_results["da_MAE_mean"]
            result["counts"] = DA_results["counts"]
            result["mse_ref"] = DA_results["mse_ref"]
            result["mse_DA"] = DA_results["mse_DA"]
            
            if self.settings.ONE_DIM:
                result["mse_ref_points"] = DA_results["mse_ref_points"]
                result["mse_da_points"] = DA_results["mse_da_points"]

                #Big Data Added Results
                result["mse_ref_overlap"] = DA_results["mse_ref_overlap"]
                result["mse_da_overlap"] = DA_results["mse_da_overlap"]
                result["mse_percent_points"] = DA_results["mse_percent_points"]
                result["mse_percent_overlap"] = DA_results["mse_percent_overlap"]
                result["time_overlap"] = DA_results["time_overlap"]
                result["time_saving_uc"] = DA_results["time_saving_uc"] 
                result["time_saving_da"] = DA_results["time_saving_da"]
                result["time_saving_u0"] = DA_results["time_saving_u0"]

            if self.reconstruction:
                result["l1_loss"] = l1.detach().cpu().numpy()
                result["l2_loss"] = l2.detach().cpu().numpy()

            result["time"] = t2 - t1
            result["time_online"] = DA_results["time_online"]
            if self.save_vtu:
                tot_DA_MAE += DA_results.get("da_MAE")
                tot_ref_MAE += DA_results.get("ref_MAE")
            #add to results list (that will become a .csv)
            results.append(result)

            #add to aggregated dict results
            totals = self.__add_result_to_totals(result, totals)

            if idx % print_every == 0 and idx > 0:
                if not print_small:
                    print("idx:", idx)
                self.__print_totals(totals, idx + 1, print_small, self.settings)
        if not print_small:
            print("------------")
        output_stat = self.__print_totals(totals, num_states, print_small, self.settings)
        if not print_small:
            print("------------")

        results_df = pd.DataFrame(results)
        stat = []
        stat.append(output_stat)
        results_avg = pd.DataFrame(stat)
        if self.save_vtu and self.settings.ONE_DIM:
            tot_DA_MAE /= num_states
            tot_ref_MAE /= num_states
            out_fp_ref = "av_ref_MAE.vtu"
            out_fp_DA = "av_da_MAE.vtu"
            saveGrid(tot_ref_MAE, self.settings, True, "ref_MAE", filename=out_fp_ref)
            saveGrid(tot_DA_MAE, self.settings, True, "DA_MAE", filename=out_fp_DA)

        #save to csv
        if self.csv_fp:
            results_df.to_csv(self.csv_fp)
            results_avg.to_csv(self.csv_avg_fp)
                
        if self.plot:
            raise NotImplementedError("plotting functionality not implemented yet")
        return results_df
        
    @staticmethod
    def get_tots(results_df):
        data = {}
        data["mse_ref"] = results_df["mse_ref"].mean()
        data["mse_DA"] = results_df["mse_DA"].mean()
        data["ref_MAE_mean"] = results_df["ref_MAE_mean"].mean()
        data["da_MAE_mean"] = results_df["da_MAE_mean"].mean()
        data["percent_improvement"] = results_df["percent_improvement"].mean()
        time = results_df["time"]
        time = time[1:] #ignore the first one as this can occud offline

        data["time"] = time.mean()
        return data

    @staticmethod
    def __add_result_to_totals(result, totals):
        for k, v in result.items():
            totals[k] += v
        return totals

    @staticmethod
    def __print_totals(totals, num_states, print_small, settings):
        if not print_small:
            for k, v in totals.items():
                print(k, "{:.2f}".format(v / num_states))
            print()
        else:
            out_str_one = "DA - - L2: {:.2f}, L1: {:.2f}, % Improve: {:.2f}%, DA_MAE: {:.2f}, mse_ref: {:.2f}, mse_DA: {:.3f}, time(s): {:.4f}s,".format(
                                                    totals["l2_loss"] / num_states,
                                                    totals["l1_loss"] / num_states,
                                                    totals["percent_improvement"] / num_states,
                                                    totals["da_MAE_mean"] / num_states,
                                                    totals["mse_ref"] / num_states,
                                                    totals["mse_DA"] / num_states,
                                                    totals["time"] / num_states)
            if settings.ONE_DIM:
                out_str_two = "\% improve_point: {:.2f}, mse_ref_points: {}, mse_da_points: {}, % improve_overlap: {:.2f}, mse_ref_overlap: {:.5f}, mse_da_overlap: {:.5f}".format(
                                                    totals["mse_percent_points"] / num_states,
                                                    totals["mse_ref_points"] / num_states,
                                                    totals["mse_da_points"] / num_states,
                                                    totals["mse_percent_overlap"] / num_states,
                                                    totals["mse_ref_overlap"] /num_states,
                                                    totals["mse_da_overlap"] / num_states)
                print(out_str_two)
            print(out_str_one)
           

            out_dict = {}
            out_dict["L2_Loss"] = totals["l2_loss"] / num_states
            out_dict["L1_Loss"] =  totals["l1_loss"] / num_states

            out_dict["ref_MAE_mean"] = totals["ref_MAE_mean"] / num_states
            out_dict["da_MAE_mean"] = totals["da_MAE_mean"] / num_states

            out_dict["percent_improvement"] = totals["percent_improvement"] / num_states
            out_dict["mse_ref"] = totals["mse_ref"] / num_states
            out_dict["mse_DA"] = totals["mse_DA"] / num_states
        
            if settings.ONE_DIM:
                out_dict["mse_percent_points"] = totals["mse_percent_points"] / num_states
                out_dict["mse_ref_points"] = totals["mse_ref_points"] / num_states
                out_dict["mse_da_points"] = totals["mse_da_points"] / num_states
                
                out_dict["mse_percent_overlap"] = totals["mse_percent_overlap"] / num_states
                out_dict["mse_ref_overlap"] = totals["mse_ref_overlap"] / num_states
                out_dict["mse_da_overlap"] = totals["mse_da_overlap"] / num_states
                time_saving_vtu = totals["time_saving_da"] + totals["time_overlap"] + totals["time_saving_uc"] + totals["time_saving_u0"]
                out_dict["real_time"] = (totals["time"] - time_saving_vtu) / num_states

            out_dict["time"] = totals["time"] / num_states #Time to run .DA_AE() or .DA_SVD() methods
            
            return out_dict
