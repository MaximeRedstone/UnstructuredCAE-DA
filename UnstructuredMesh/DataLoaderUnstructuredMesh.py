
""" Data Loader for CAEs on Unstructured Meshes """
import sys, os, pickle, argparse

import numpy as np
from tabulate import tabulate
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist

from UnstructuredCAEDA.train import TrainAE
from UnstructuredCAEDA.VarDA import BatchDA
from UnstructuredCAEDA.data import GetData
from UnstructuredCAEDA.settings.models.CLIC import CLIC
from UnstructuredCAEDA.fluidity import vtktools

from UnstructuredCAEDA.UnstructuredMesh.Localisation import *
from UnstructuredCAEDA.UnstructuredMesh.HelpersUnstructuredMesh import *

class DataLoaderUnstructuredMesh(GetData):
    """ Class to load data from data files (.vtu) for AEs Training or Data Assimilation 
    Abreviations: fp = file path 
                  v = vertices
                  f = faces
                  ug = unstructured grid """

    def __init__(self):
        pass
    
    def get_X(self, settings):
        """ Args: settings: (CLICUnstructuredMesh class)
            Returns: np.array of dimensions time steps x localised points """
        fps = DataLoaderUnstructuredMesh.get_sorted_fps_U(settings.getDataDir())
        filepath = getXFilePath(settings)
        filepathIdxMatch = getMeshToNetworkIdxMatchingLocations(settings)
        filepathIdxAll = getMeshToNetworkIdxLocations(settings)
        filepathIdxOverlap = getIdxOverlap(settings)
        if os.path.isfile(filepath):
            print("Reading pickle file: ", filepath)
            with open(filepath, "rb") as f:
                X = pickle.load(f)
            if settings.CLIC_UNSTRUCTURED: #Only read if localisation took place first
                with open(filepathIdxMatch, "rb") as f:
                    idxDict = pickle.load(f)
                    settings.setMatchIdxMeshToIdxNetwork(idxDict)
                with open(filepathIdxAll, "rb") as f:
                    idxDict = pickle.load(f)
                    settings.setIdxMeshToIdxNetwork(idxDict)
                with open(filepathIdxOverlap, "rb") as f:
                    idxOverlap = pickle.load(f)
                    settings.setOverlappingRegions(idxOverlap)
        else:
            print("Creating pickle file for percentage: ", settings.getPercentOfVertices())
            X = Localiser.createX(fps, settings)
            outfile = open(filepath, 'wb')
            pickle.dump(X, outfile)
            outfile.close()

        print("Shape read X: ", np.shape(X))
        networkInput= DataLoaderUnstructuredMesh.toNetworkSpace(X, settings)
        return networkInput

    @staticmethod
    def get_sorted_fps_U(data_dir):
        """ Creates and returns list of .vtu filepaths sorted according
        to timestamp in name.
        Input files in data_dir must be of the
        form LSBU_<TIMESTEP INDEX>_<SUBDOMAIN>.vtu """
        
        fps = [f for f in os.listdir(data_dir) if not f.startswith('.')]

        #Extract Subdomain number from data_dir
        _, subdomain = data_dir.split("_") #split returns ["LSBU", "<Subdomain>"]
        subdomain.replace("/", "")

        #Extract index of timestep from file name
        idx_fps = []
        for fp in fps:
            if not fp.startswith("."):
                lsbu, timestep, extension = fp.split("_")
                idx = int(timestep)
                idx_fps.append(idx)

        #sort by timestep
        assert len(idx_fps) == len(fps)
        zipped_pairs = zip(idx_fps, fps)
        fps_sorted = [x for _, x in sorted(zipped_pairs)]

        #add absolute path
        fps_sorted = [data_dir + x for x in fps_sorted]
        return fps_sorted

    @staticmethod
    def toNetworkSpace(X, settings):
        """ Convert X to appropriate shape for AEs """
        if settings.getDim() == 1:
            networkInput = X
        elif settings.getDim() == 2:
            networkInput = DataLoaderUnstructuredMesh.reshape(X)
        else:
            raise NotImplementedError("Converting to network space only works for specific cases (see function description)")

        return networkInput

    # DEPRECATED CODE
    # Below are functions used when experimenting with Tucodec 2D model that required minimum dimensions 
    # of 90 x 92 for size of convolution kernels to be smaller than data input size at all layers
    @staticmethod
    def reshape(X):
        """ Reshape X of shape Timesteps x Scalars to Timesteps x nx x ny """
        nbTimeSteps, nbVertices = np.shape(X)[0], np.shape(X)[1]
        newshape, idxEnd = DataLoaderUnstructuredMesh.setNetworkSpace(nbVertices, np.ndim(X))
        print("X entering network reshaping len = {} when will become = {}".format(nbVertices, idxEnd))
        Xidx = []
        for timestep in X:
            timestep = timestep[:idxEnd]
            Xidx.append(timestep)

        for idx in range(nbTimeSteps):
            oneTimestep = Xidx[idx]
            oneTimestepReshaped = np.reshape(oneTimestep, newshape, order='F')
            if idx == 0:
                #fix length of vectors and initialize the output array:
                nsize = np.shape(oneTimestepReshaped)
                size = (nbTimeSteps,) + nsize
                output = np.zeros(size)
            output[idx] = oneTimestepReshaped
            
        print("In network space, Xreshaped = ", np.shape(output))
        return output

    @staticmethod
    def setNetworkSpace(n, dim):
        if dim not in [2]:
           raise NotImplementedError("function can only reshape 2D inputs")
 
        nbOfVertices =  int(n - (n % 10))
        x, y = 92, 90
        x, y = DataLoaderUnstructuredMesh.getXfromY(nbOfVertices, y)
        newshape = x, y
        idxEnd = x * y
        return newshape, idxEnd

    
    @staticmethod
    def getXfromY(n, ymin):
        xres = n // ymin
        if (xres % 2 != 0):
            xres = xres - 1
        return xres, ymin