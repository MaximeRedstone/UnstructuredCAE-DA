from tabulate import tabulate
from UnstructuredCAEDA.fluidity import vtktools
import numpy as np
import torch
from torch import nn
import os, sys

def getOtherSubdomainsFromFp(allDataDir, thisDomain):
    """ Returns array of other sub-domains considered in Big Data Assimilation """
    otherDomains = []
    for idx in range(len(allDataDir)):
        if idx != thisDomain:
            other = getSubdomainFromFP(allDataDir[idx])
            otherDomains.append(other)
    return otherDomains

def getThisDomainPickleDir(pickledDir, thisDomain):
    """ Returns pickle directory where data is stored (if already pickled) for this particular sub-domain """
    return pickledDir + "X_" + str(thisDomain) + "/"

def getSubdomainFromFP(path):
    """ Returns sub-domain number """
    _, domain = path.split("subdomain_")
    return domain[:-1]

def saveGrid(u, settings, avg, label, filename=None, trainSteps=None, idx=None):
    """ Save results of DA in .vtu file 
    As localisation only works with relevants locations in original domain, 
    we take original subdomain as baseline and replace 
    relevant locations scalar values post assimilation to produce this results grid """
    try:
        os.chdir(os.getcwd() + str(settings.getExptDir()))
    except OSError:
        pass
        
    fileDir = "vtu"
    if not os.path.exists(fileDir):
        os.mkdir(fileDir)

    if avg:
        saveFilePath = fileDir + "/t" + str(settings.DA_BACKGROUND_IDX) + "_" + str(filename)
        originalGridFP = getDataPath(settings.DATA_FP, settings.DA_BACKGROUND_IDX)
    else:
        timestep = trainSteps + idx
        fileDir = fileDir + "/" + str(timestep) + "/"
        if not os.path.exists(fileDir):
            os.mkdir(fileDir) 
        filename = str(label) + ".vtu"
        saveFilePath = fileDir + filename
        originalGridFP = getDataPath(settings.DATA_FP, timestep)

    ug = vtktools.vtu(originalGridFP)
    uDA = incorporateResults(u, ug, settings)
    ug.AddScalarField(str(label), uDA)
    ug.Write(saveFilePath)

def incorporateResults(u, ug, settings):
    """ Using one unstructured grid as baseline, replaces scalar values of relevant locations post assimilation to produce updated grid """
    scalar = ug.GetScalarField('TracerGeorge')
    for idxNet, element in enumerate(u.flatten()):
        idxMesh = getKey(settings.IDX_MESH_TO_IDX_NETWORK, idxNet)
        if idxMesh != -1:
            scalar[idxMesh] = element
    return scalar

def getSettingsInfo(settings):
    """ Recap of Settings used to conduct an experiment """
    print(tabulate([
    ['Experiment title', settings.TITLE],    
    ['Home dir', settings.HOME_DIR],
    ['Data fp', settings.DATA_FP],
    ['Pickled data fp', settings.X_PICKLE],
    ['Field name', settings.FIELD_NAME],
    ['Using Loader', settings.LOADER],
    ['Compression Method', settings.COMPRESSION_METHOD],
    ['Percentage', settings.PERCENTAGE],
    ['Seed', settings.SEED],
    ['3D', settings.THREE_DIM],
    ['2D', settings.TWO_DIM],
    ['1D', settings.ONE_DIM],
    ['Observational Fraction', settings.OBS_FRAC],
    ['Observational Variance', settings.OBS_VARIANCE],
    ['DA Background State as Mean of Historical Data', settings.MEAN_HIST_DATA]],
    headers=['Header', 'Info'], tablefmt='orgtbl'))

def getXFilePath(settings):
    """ Find file path of saved X data in different cases """
    if settings.CLUSTERING and settings.getDim() == 1:
        filename = "X_" + str(settings.getPercentOfVertices()) + "_1D_clustered.pickle"
    elif settings.CLUSTERING and settings.getDim() == 2:
        filename = "X_" + str(settings.getPercentOfVertices()) + "_2D_clustered.pickle"
    else:
        print("--- No clustering Procedure ---")
        filename = "X_" + str(settings.getPercentOfVertices()) + ".pickle"
    filepath = str(settings.X_PICKLE) + filename
    return filepath

def getMeshToNetworkIdxMatchingLocations(settings):
    """ Returns file path of saved dictionary associating location's idx in unstructured mesh with 
    its idx in array of clustered data (for locations present in all sub-domains, e.g. overlapping regions)"""
    filename = "meshIdxToNetIdxMatching_" + str(settings.getDim())+ "D.pickle"
    filepath = str(settings.X_PICKLE) + filename
    return filepath

def getMeshToNetworkIdxLocations(settings):
    """ Returns file path of saved dictionary associating location's idx in unstructured mesh with its idx in array of clustered data """
    filename = "meshIdxToNetIdx_" + str(settings.getDim()) + "D.pickle"
    filepath = str(settings.X_PICKLE) + filename
    return filepath

def getIdxOverlap(settings):
    """ Returns file path, for a particular sub-domain, to overlapping location's indices in the unstructured mesh """
    filename = "idxOverlap_" + str(settings.getDim()) + "D.pickle"
    filepath = str(settings.X_PICKLE) + filename
    return filepath

def getDataPath(directory, timestep):
    """ Returns file path to vtu file for a particular timestep """
    _, domain = directory.split("subdomain_")
    domain = domain[:-1]
    filepath = directory + "LSBU_" + str(timestep) + "_" + str(domain) + ".vtu"
    return filepath

def mkdir(mypath):
    """ Creates a directory. equivalent to using mkdir -p on the command line """
    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

def getKey(myDict, value):
    """ Finds key associated with a value in dictionary """
    for key, val in myDict.items():
        if val == value:
            return key
    return -1

def printScalar(clusters):
    for idx, cluster in enumerate(clusters):
        print("Cluster: ", idx, " \n")
        print(len(cluster), " x ", len(cluster[0]))
        print("\n----\n")

def getDomainFromFP(fp):
    """ Returns domain number from file path """
    path, fileInfo = fp.split("LSBU_") #split returns ["/user/.../data/subdomain_8/LSBU", "<timestep>_<subdomain>.vtu"]
    timestep, domain = fileInfo.split("_")
    return domain[:-4] #deletes .vtu extension

def getNumberOfVertices(dataPath, percentOfVertices):
    """ Returns number of vertices equivalent to percentages of locations required by user """
    ug = vtktools.vtu(dataPath)
    locations = ug.GetLocations()
    return len(locations) * percentOfVertices / 100

# Case of 1D arrays 
def matchDimensions1DVarDA(arr1, arr2):
    """ Finds difference in dimensions between two arrays and pads smallest to allow for mathematical operations """
    difference = findDifferenceOneAxis(arr1, arr2, 1)
    absDifference = np.absolute(difference)
    if difference > 0:
        arr2 = np.pad(arr2, (0,absDifference), 'reflect')
        return arr1, arr2
    elif difference < 0:
        arr1 = np.pad(arr1, (0, absDifference), 'reflect')
        return arr1, arr2
    return arr1, arr2

def matchDimensions1D(trunk, mask):
    """ Finds difference in dimensions between two matrices and pads smallest to allow for matrix operations """
    difference = findDifferenceOneAxis(trunk, mask, 1)
    left, right = findPaddingBothSidesOneAxis(np.absolute(difference))

    if difference > 0:
        pad = nn.ReflectionPad1d((left, right))
        mask = pad(mask)
        return trunk, mask
    elif difference < 0:
        pad = nn.ReflectionPad1d((left, right))
        trunk = pad(trunk)
        return trunk, mask
    return trunk, mask

def findDifferenceOneAxis(trunk, mask, axis):
    """ Returns difference along one axis (x or y) """
    lenTrunk = len(trunk.shape) - axis
    lenMask = len(mask.shape) - axis
    difference = trunk.shape[lenTrunk] - mask.shape[lenMask]
    return difference

def findPaddingBothSidesOneAxis(difference):
    """ Returns amount of padding required on both sides based on difference found """
    if difference % 2 == 0:
        left, right = int(difference/2), int(difference/2)
    else:
        left, right = int(difference/2), int(difference/2 + difference%2)
    return left, right

def findDifferenceCombination(differenceX, differenceY):
    if differenceX == 0 and differenceY == 0:
        return 0
    elif differenceX > 0:
        if differenceY > 0:
            return 1
        elif differenceY < 0:
            return 2
        else:
            return 3
    elif differenceX < 0:
        if differenceY > 0:
            return 4
        elif differenceY < 0:
            return 5
        else:
            return 6
    elif differenceX == 0:
        if differenceY > 0:
            return 7
        elif differenceY < 0:
            return 8

# Case of 2D matrices
def matchDimensions2DVarDA(arr1, arr2):
    """ Finds difference in dimensions between two matrices and pads smallest to allow for mathematical operations """
    differenceX = findDifferenceOneAxis(arr1, arr2, 2)
    differenceY = findDifferenceOneAxis(arr1, arr2, 1)
    leftX, rightX = findPaddingBothSidesOneAxis(np.absolute(differenceX))
    bottomY, topY = findPaddingBothSidesOneAxis(np.absolute(differenceY))

    if differenceX == 0 and differenceY == 0: #same size
        return arr1, arr2

    elif differenceX > 0: #Difference in X
        if differenceY > 0: # and Y
            arr2 = np.pad(arr2, ((leftX, rightX), (bottomY, topY)), 'reflect')
            return arr1, arr2
        elif differenceY < 0:
            arr2 = np.pad(arr2, ((leftX, rightX), (0, 0)), 'reflect')
            arr1 = np.pad(arr1, ((0, 0), (bottomY, topY)), 'reflect')
            return arr1, arr2
        else: # only X
            arr2 = np.pad(arr2, ((leftX, rightX), (0, 0)), 'reflect')
            return arr1, arr2

    elif differenceX < 0: #Difference in X 
        if differenceY > 0: # and Y
            arr1 = np.pad(arr1, ((leftX, rightX), (0, 0)), 'reflect')
            arr2 = np.pad(arr2, ((0, 0), (topY, bottomY)), 'reflect')
            return arr1, arr2
        elif differenceY < 0:
            arr1 = np.pad(arr1, ((leftX, rightX), (bottomY, topY)), 'reflect')
            return arr1, arr2
        else: # only X
            arr1 = np.pad(arr1, ((leftX, rightX), (0, 0)), 'reflect')
            return arr1, arr2

    elif differenceX == 0: #Only difference in Y
        if differenceY > 0:
            arr2 = np.pad(arr2, ((0, 0), (bottomY, topY)), 'reflect')
            return arr1, arr2
        elif differenceY < 0:
            arr1 = np.pad(arr1, ((0, 0), (bottomY, topY)), 'reflect')
            return arr1, arr2

    return arr1, arr2

def matchDimensions2D(trunk, mask, RABMatching=False, DAProcess=False):
    """ Finds difference in dimensions between two matrices and pads smallest to allow for mathematical operations (works both in case of DA or RAB blocks)"""
    differenceX = findDifferenceOneAxis(trunk, mask, 2)
    differenceY = findDifferenceOneAxis(trunk, mask, 1)

    if RABMatching:
        trunkValuesX, trunkValuesY, maskValuesX, maskValuesY = trunk[0][0][0][:], trunk[0][0][:][0], mask[0][0][0][:], mask[0][0][:][0]
    elif DAProcess:
        trunkValues, maskValues = trunk, mask
    else:
        trunkValues, maskValues = trunk[0][0], mask[0]

    leftX, rightX = findPaddingBothSidesOneAxis(np.absolute(differenceX))
    bottomY, topY = findPaddingBothSidesOneAxis(np.absolute(differenceY))

    if DAProcess:
        mask = (mask.unsqueeze(0)).unsqueeze(0)
        trunk = (trunk.unsqueeze(0)).unsqueeze(0)

    if differenceX == 0 and differenceY == 0:
        return trunk, mask

    elif differenceX > 0:
        if differenceY > 0:
            pad = nn.ReflectionPad2d((topY, bottomY, leftX, rightX))
            mask = pad(mask)
            return trunk, mask
        elif differenceY < 0:
            padMask = nn.ReflectionPad2d((0, 0, leftX, rightX))
            padTrunk = nn.ReflectionPad2d((topY, bottomY, 0, 0))
            mask = padMask(mask)
            trunk = padTrunk(trunk)
            return trunk, mask
        else:
            pad = nn.ReflectionPad2d((0, 0, leftX, rightX))
            mask = pad(mask)
            return trunk, mask

    elif differenceX < 0:
        if differenceY > 0:
            padTrunk = nn.ReflectionPad2d((0, 0, leftX, rightX))
            padMask = nn.ReflectionPad2d((topY, bottomY, 0, 0))
            mask = padMask(mask)
            trunk = padTrunk(trunk)
            return trunk, mask
        elif differenceY < 0:
            pad = nn.ReflectionPad2d((topY, bottomY, leftX, rightX))
            trunk = pad(trunk)
            return trunk, mask
        else:
            pad = nn.ReflectionPad2d((0, 0, leftX, rightX))
            trunk = pad(trunk)
            return trunk, mask

    elif differenceX == 0:
        if differenceY > 0:
            padMask = nn.ReflectionPad2d((topY, bottomY, 0, 0))
            mask = padMask(mask)
            return trunk, mask
        elif differenceY < 0:
            padTrunk = nn.ReflectionPad2d((topY, bottomY, 0, 0))
            trunk = padTrunk(trunk)
            return trunk, mask

    return trunk, mask

