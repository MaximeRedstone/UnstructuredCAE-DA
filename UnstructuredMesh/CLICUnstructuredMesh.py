
""" Unstructured Mesh implementation of the CLIC class used for Convolutional Autoencoders on structured meshes."""

from UnstructuredCAEDA.settings.models.CLIC import CLIC
from UnstructuredCAEDA.UnstructuredMesh.DataLoaderUnstructuredMesh import *

class CLICUnstructuredMesh(CLIC):
    def __init__(self, CLIC_kwargs, opt_kwargs):
        super().__init__(**CLIC_kwargs)

        self.PERCENTAGE = opt_kwargs["percentage"]
        self.TWO_DIM = opt_kwargs["2D"]
        self.ONE_DIM = opt_kwargs["1D"]
        self.THIS_DOMAIN = opt_kwargs["thisDomain"]
        self.OTHER_DOMAINS = opt_kwargs["otherDomains"]
        self.DA_BACKGROUND_IDX = opt_kwargs["daBackgroundIdx"]
        self.COMPRESSION_METHOD = opt_kwargs["compressionMethod"]
        self.REDUCED_SPACE = opt_kwargs["reducedSpace"]
        self.MEAN_HIST_DATA = opt_kwargs["backgroundIsMeanHistoricalData"]
        self.OBS_VARIANCE = opt_kwargs["obsVariance"]
        self.NAME = opt_kwargs["modelName"]

        self.THREE_DIM = False #Override base class
        self.DEBUG = False
        self.SHUFFLE_DATA = False
        self.SAVE = True
        self.NORMALIZE = True
        self.UNDO_NORMALIZE = True

        self.DATA_FP = None
        self.X_PICKLE = None
        self.INTERMEDIATE_FP = None
        self.TITLE = "No Title given"

        self.MIN_DIM = 1024 # max size of first linear layer
        
        # Big data Assimilation
        self.CLUSTERING = True
        self.OVERLAPPING_REGIONS = None # Find indices of overlapping regions amongst sub-domains
        self.IDX_MESH_TO_IDX_NETWORK = None # Keep track of location idx in mesh and location idx in clustered data (array of points)
        self.IDX_MESH_TO_IDX_NETWORK_MATCH = None # for overlapping regions
        self.EXPT_DIR = None
    
        self.CLIC_UNSTRUCTURED = True

        # DA parameters
        self.OBS_FRAC = 0.1 # Using 1 = All observations taken into account for SVD 
        self.NUMBER_MODES = None # Set using optimal truncation parameter algorithm from https://www.sciencedirect.com/science/article/pii/S0021999118307095
        
    def get_number_modes(self):
        return self.NUMBER_MODES

    def setIntermediatePath(self, path):
        self.INTERMEDIATE_FP = path

    def setExptDir(self, path):
        self.EXPT_DIR = path
        
    def getExptDir(self):
        return self.EXPT_DIR
        
    def setDataPath(self, path):
        self.DATA_FP = path

    def getDataPath(self):
        return self.DATA_FP
        
    def setThisDomain(self, domain):
        self.THIS_DOMAIN = domain

    def setOtherDomain(self, domains):
        self.OTHER_DOMAINS = domains
    
    def setOverlappingRegions(self, indices):
        self.OVERLAPPING_REGIONS = indices
    
    def setIdxMeshToIdxNetwork(self, idxDict):
        self.IDX_MESH_TO_IDX_NETWORK = idxDict

    def setMatchIdxMeshToIdxNetwork(self, idxDict):
        self.IDX_MESH_TO_IDX_NETWORK_MATCH = idxDict

    def getOverlappingRegions(self):
        return self.OVERLAPPING_REGIONS

    def getThisSubdomainFP(self):
        path = self.DATA_FP
        pathDir, _ = path.split("subdomain_")
        fp = pathDir + "subdomain_" + str(self.THIS_DOMAIN) + "/"
        fp = fp + "LSBU_0_" + str(self.THIS_DOMAIN) + ".vtu"
        return fp

    def getOtherSubdomainFP(self):
        path = self.DATA_FP
        pathDir, _ = path.split("subdomain_")
        if len(self.OTHER_DOMAINS) != 1:
            raise NotImplementedError("Not Implemented more than 2 subdomains in Big Data Assimilation")
        else:
            fp = pathDir + "subdomain_" + str(self.OTHER_DOMAINS[0]) + "/"
            fp = fp + "LSBU_0_" + str(self.OTHER_DOMAINS[0]) + ".vtu"
        return fp

    def getSubdomain(self):
        path = self.DATA_FP
        _, domain = path.split("subdomain_")
        return domain[:-1]

    def setXPickledPath(self, path, newDomain=None):
        if newDomain != None:
            pathDir, _ = path.split("X_")
            newPath = pathDir + "X_" + str(newDomain) + "/"
            self.X_PICKLE = newPath
        else:
            self.X_PICKLE = path

    def setExpTitle(self, title):
        self.TITLE = title
    
    def setClustering(self, clustering):
        self.CLUSTERING = clustering

    def setResultsPath(self, path):
        self.RESULTS_FP = path
        
    def getExpTitle(self):
        return self.TITLE

    def getPercentOfVertices(self):
        return self.PERCENTAGE

    def getDim(self):
        if self.ONE_DIM:
            return 1
        elif self.TWO_DIM:
            return 2
        elif self.THREE_DIM:
            return 3
        else:
            raise NotImplementedError("Dim of data must be 1, 2 or 3 Dimensions")
        return 0
    
    def getMinDim(self):
        return self.MIN_DIM

    def findOverlappingRegions(self):
        allLocations = DataLoaderUnstructuredMesh.getAllLocations(self.getThisSubdomainFP())
        allLocationsOtherSubdomain = DataLoaderUnstructuredMesh.getAllLocations(self.getOtherSubdomainFP())
        matchingInformation = DataLoaderUnstructuredMesh.findMatchingLocations(allLocations, allLocationsOtherSubdomain)
        self.OVERLAPPING_REGIONS = matchingInformation["idx1"]
