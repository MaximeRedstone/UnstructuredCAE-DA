
""" Localiser to implement Localisation by Clustering for CAEs on Unstructured Meshes """
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

from UnstructuredCAEDA.UnstructuredMesh.HelpersUnstructuredMesh import *

class Localiser():
    """ Class to run localisation by clustering on .vtu files """
    def __init__(self):
        pass

    @staticmethod
    def normaliseScalar(scalar):
        """ Normalise a scalar field between 0 and 1 """
        normalisedScalar = []
        maxValue = np.amax(scalar)
        minValue = np.amin(scalar)
        for value in scalar:
            normalisedValue = (value - minValue) / (maxValue - minValue)
            normalisedScalar.append(normalisedValue)
        return normalisedScalar 

    @staticmethod
    def selectLocations(dataPath, percentOfVertices, locations, tracer='TracerGeorge'):
        """ Select Most X% relevant locations (i.e. that have the most information about tracer) from one timestep"""
        ug = vtktools.vtu(dataPath)
        sfield = ug.GetScalarField(tracer)
        normalisedSField = Localiser.normaliseScalar(sfield)
        percentile = np.percentile(normalisedSField, 100-percentOfVertices) #Computes (100-percentRequired)% percentile
        relevantLocations = []
        for idx in range(len(locations)):
            scalar = normalisedSField[idx]
            if scalar > percentile:
                relevantLocations.append(idx)
        return relevantLocations
        
    @staticmethod
    def findClusters(locationsInAllTimesteps, allLocations, outputdir):
        """ From list of relevant locations (indices) occuring in all timesteps,
         group them by cluster (i.e. conserving mesh spatial information) using DBSCAN procedure """
        locationCoordinates = []
        for idx in locationsInAllTimesteps:
            locationCoordinates.append(allLocations[idx].tolist())

        locationCoordinates = np.array(locationCoordinates)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(locationCoordinates[:,0], locationCoordinates[:,1], locationCoordinates[:,2], s=5)
        ax.view_init(azim=200)
        mkdir(outputdir)
        plt.savefig('{}/clusters_input.png'.format(outputdir))

        rangeOfEps = [5, 10, 20, 15]
        print("Entering DBSCAN at {}".format(datetime.now()))
        for distance in rangeOfEps:
            model = DBSCAN(eps=distance, min_samples=100) #Choice of eps matters as represents distance (too big will merge clusters together
            model.fit_predict(locationCoordinates) #, too small yields lots of point no being assigned to a cluster)
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(locationCoordinates[:,0], locationCoordinates[:,1], locationCoordinates[:,2], c=model.labels_, s=5)
            ax.view_init(azim=200)
            plt.savefig('{}/clusters_results_EPS_{}.png'.format(outputdir, distance))
        print("DBSCAN Finished at {}".format(datetime.now()))

        print("number of cluster found: {}".format(len(set(model.labels_))))
        print('cluster for each point: ', model.labels_)

        clusters = []
        clusterIds = list(dict.fromkeys(model.labels_))
        print("Cluster ids:", clusterIds)
        for clusterId in clusterIds:
            count = 0
            clusterI = []
            for label in model.labels_:
                if clusterId == label:
                    clusterI.append(locationsInAllTimesteps[count])
                count += 1

            if clusterId != -1: #Do not return list of points that do not belong to a cluster (default value -1 given by DBSCAN)
                clusters.append(clusterI)
        
        print("Cluster found: {}".format(len(clusters)))
        count = 0
        for cluster in clusters:
            print("Cluster {} has {} locations".format(count, len(cluster)))
            count += 1
        return clusters

    @staticmethod
    def localisationWorked(relevantLocations, nbOfVertices, allLocations, outputdir, clustering):
        """ Check selected locations are included in all timesteps to provide valid input for CAEs (same spatial inputs) """
        locationsInAllTimesteps = []
        firstTimeStep = relevantLocations[0]
        for idx in firstTimeStep:
            inAllOtherTimeSteps = True
            for otherTimeStep in relevantLocations:
                if idx not in otherTimeStep:
                    inAllOtherTimeSteps = False
                    break
            if inAllOtherTimeSteps:
                locationsInAllTimesteps.append(idx)

        locationsInAllTimesteps = list(dict.fromkeys(locationsInAllTimesteps))

        if len(locationsInAllTimesteps) >= nbOfVertices:
            if clustering:
                clustersInAllTimesteps = Localiser.findClusters(locationsInAllTimesteps, allLocations, outputdir)
                return True, clustersInAllTimesteps
            else:
                return True, locationsInAllTimesteps

        return False, locationsInAllTimesteps

    @staticmethod
    def localise(fps, settings):
        """ Attempts to find nbOfVertices locations that are contained in all timesteps """
        relevantLocations = []
        orderedLocationsInAllTimesteps = []
        lookingForLocalisation = True
        percentOfVertices, clustering, minShape = settings.PERCENTAGE, settings.CLUSTERING, settings.getMinDim()
        domainName = getDomainFromFP(fps[0])
        nbOfVertices = getNumberOfVertices(fps[0], percentOfVertices)
        nbOfVertices =  int(nbOfVertices - (nbOfVertices % 10))
        print("REQUIREMENTS: {}% of locations corresponds to {} vertices.".format(percentOfVertices, nbOfVertices))

        percentOfVertices = 5 + percentOfVertices
        nbOfVerticesPool = getNumberOfVertices(fps[0], percentOfVertices)
        nbOfVerticesPool =  int(nbOfVerticesPool - (nbOfVerticesPool % 10))
        assert nbOfVerticesPool > minShape, "Tucodec Model Layer Architectures (1D2L, 1D4L, 1D0L) requires min dimension of {} points".format(minShape)

        allLocations = Localiser.getAllLocations(fps[0])
        allLocationsOtherSubdomain = Localiser.getAllLocations(settings.getOtherSubdomainFP())
        matchingInformation = Localiser.findMatchingLocations(allLocations, allLocationsOtherSubdomain)
        settings.setOverlappingRegions(matchingInformation["idx1"])
        while lookingForLocalisation:
            print("Attempted at {} ".format(datetime.now()))
            for dataPath in fps:
                relevantLocationsPerTimestep = Localiser.selectLocations(dataPath, percentOfVertices, allLocations)
                relevantLocations.append(relevantLocationsPerTimestep)
            
            print("Selected relevant locations for each timestep {} ".format(datetime.now()))
            outputdir = 'LocalisationResults/subdomain_' + str(domainName) + '_' + str(percentOfVertices) + 'PercentOfData'
            foundLocalisation, locationsInAllTimesteps = Localiser.localisationWorked(relevantLocations, nbOfVertices, allLocations, outputdir, clustering)
            if not foundLocalisation:
                print("Localisation Failed - {}%% of data not enough".format(percentOfVertices))
                print("Flushing Array... Increasing Pool of locations (Percentage + 5%) at {}".format(datetime.now()))
                relevantLocations = []
                percentOfVertices = percentOfVertices + 5
            else:
                print("Localisation Success! Percentage retained {} at {}".format(percentOfVertices, datetime.now()))
                print("Length of clustering: ", len(locationsInAllTimesteps))
                print("Defining Mesh idx to Network idx started at: ",datetime.now())
                orderedLocationsInAllTimesteps, meshIdxToNetworkIdxMatching, meshIdxToNetworkIdx = Localiser.insertMatchingLocations(matchingInformation, allLocations, locationsInAllTimesteps)
                print("Defining Mesh idx to Network idx ended at: ",datetime.now())
                settings.setMatchIdxMeshToIdxNetwork(meshIdxToNetworkIdxMatching)
                settings.setIdxMeshToIdxNetwork(meshIdxToNetworkIdx)
                outfile = open(getMeshToNetworkIdxMatchingLocations(settings), 'wb')
                pickle.dump(meshIdxToNetworkIdxMatching, outfile)
                outfile.close()
                outfile = open(getMeshToNetworkIdxLocations(settings), 'wb')
                pickle.dump(meshIdxToNetworkIdx, outfile)
                outfile.close()
                outfile = open(getIdxOverlap(settings), "wb")
                pickle.dump(matchingInformation['idx1'], outfile)
                outfile.close()
                print("Length of clustering + matching locations = {} at {} ".format(len(orderedLocationsInAllTimesteps), datetime.now()))
                return orderedLocationsInAllTimesteps
        
        return orderedLocationsInAllTimesteps

    @staticmethod
    def getAllLocations(fp):
        """ Returns all locations from filepath to vtu file """
        ug = vtktools.vtu(fp)
        locations = ug.GetLocations()
        return locations

    @staticmethod
    def findMatchingLocations(locationsI, locationsJ):
        """ Returns information about overlapping regions (or matching locations between two sets of locations) 
        idx1: indices of matching locations in first set
        idx2: indices of matching locations in second set 
        coordinates: x,y,z coordinates of matching locations """
        print("Finding matching locations started at: ",datetime.now())
        matchingLocations, idxI, idxJ, idxToRemove = [], [], [], []
        for idx, loc in enumerate(locationsI):
            if (locationsJ == loc).all(axis=1).any():
                matchingLocations.append(loc)
                idxI.append(idx)
                idxJ.append(np.where(loc == locationsJ))
        
        for idx in range(len(idxI)):
            if len(locationsI[idxI[idx]]) != 3 or len(locationsJ[idxJ[idx]]) != 3:
                idxToRemove.append(idx)

        for i in sorted(idxToRemove, reverse=True):
            idxI.pop(i)
            idxJ.pop(i)

        matchingInformation = {"coordinates": matchingLocations,
                                "idx1": idxI,
                                "idx2": idxJ}
        print("Finding matching locations ended at: ",datetime.now())
        return matchingInformation

    @staticmethod
    def insertMatchingLocations(matchingInformation, allLocations, locationsInAllTimesteps):
        """ Inserts matching locations in relevant locations present in all time steps to compare performance of each sub-domains' CAEs on overlapping regions """
        idxDomain1 = matchingInformation["idx1"]   
            
        meshIdxToNetworkIdxMatching = {}
        meshIdxToNetworkIdx = {}
        print("Inserting matching locations in relevant locations all timesteps started at: ",datetime.now())
        if len(locationsInAllTimesteps) != 0:
            locationsInsertedAllTimesteps = np.hstack(locationsInAllTimesteps)
            for element in idxDomain1:
                coordinates = Localiser.idxToCoordinates(allLocations, locationsInsertedAllTimesteps)
                coordinatesToAdd = [allLocations[element]]
                idxToInsert, distance = Localiser.getClosestPoints(coordinatesToAdd, coordinates)
                if np.isclose(distance, 0): #disgard if location already exists in relevant locations present in all timesteps
                    pass
                else:
                    locationsInsertedAllTimesteps = np.insert(locationsInsertedAllTimesteps, idxToInsert[1][0], element)
            print("Inserting matching locations in relevant locations all timesteps ended at: ",datetime.now())
            print("Creating Dict Matching locations started at: ",datetime.now())
            for element in idxDomain1:
                index = np.where(locationsInsertedAllTimesteps == element)
                meshIdxToNetworkIdxMatching[element] = index[0][0]
            print("Creating Dict Matching locations ended at: ",datetime.now())
            print("Creating Dict locations started at: ",datetime.now())
            for idx, element in enumerate(locationsInsertedAllTimesteps):
                meshIdxToNetworkIdx[element] = idx
            print("Creating Dict locations ended at: ",datetime.now())
        else:
            # No relevant locations present in all timesteps
            locationsInsertedAllTimesteps = idxDomain1
            for idx, element in enumerate(locationsInsertedAllTimesteps):
                meshIdxToNetworkIdx[element] = idx 
            meshIdxToNetworkIdxMatching = meshIdxToNetworkIdx

        return locationsInsertedAllTimesteps, meshIdxToNetworkIdxMatching, meshIdxToNetworkIdx

    @staticmethod
    def getClosestPoints(locationToAdd, locations):
        """ Returns optimal index to place a location in array of locations based on 3D coordinates to preserve spatial information """
        distances = cdist(locationToAdd, locations)
        return np.where(distances == distances.min()), distances.min()

    @staticmethod
    def idxToCoordinates(allLocations, indices):
        """ Find coordinates of locations based on idx """
        locations = []
        for idx in indices:
            locations.append(allLocations[idx])
        return locations

    @staticmethod
    def minCluster(clusters):
        """ Returns cluster with minimum points """
        minLen = sys.maxsize
        for cluster in clusters:
            clusterLen = len(cluster)
            if clusterLen < minLen:
                minLen = clusterLen
        return minLen

    @staticmethod   
    def clustersToScalar(fps, locationsIdx, tracer='TracerGeorge'):
        """ Converts clustered locations array to array of corresponding scalar for given tracer based on idx """
        print("Converting locations to scalar started at: ",datetime.now())
        scalars = []
        for dataPath in fps:
            ug = vtktools.vtu(dataPath)
            sfield = ug.GetScalarField(tracer)
            scalarI = []
            for idx in locationsIdx:
                scalarI.append(sfield[idx])
            scalars.append(scalarI)
        print("Converting locations to scalar ended at: ",datetime.now())
        return scalars #Shape: timesteps x scalars
    
    @staticmethod
    def locationsToScalar(fps, locationsIdx, tracer='TracerGeorge'):
        """ Converts locations array to array of corresponding scalar for given tracer based on idx """
        scalars = []
        for dataPath in fps:
            ug = vtktools.vtu(dataPath)
            sfield = ug.GetScalarField(tracer)
            scalarI = []
            for idx in locationsIdx:
                scalarI.append(sfield[idx])
            scalars.append(scalarI)
        return scalars

    @staticmethod
    def createX(fps, settings):
        """ Returns array of scalars used as input for CAEs """
        orderedLocationAllTimesteps = Localiser.localise(fps, settings)
        if settings.CLUSTERING:
            scalars = Localiser.clustersToScalar(fps, orderedLocationAllTimesteps)
        else:
            scalars = Localiser.locationsToScalar(fps, orderedLocationAllTimesteps)
        return scalars