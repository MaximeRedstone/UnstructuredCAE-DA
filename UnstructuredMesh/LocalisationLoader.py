"""
Localisation Settings to conduct Localisation by Clustering before CAE training and DA.
"""

class LoaderSetting():
    def __init__(self, data, pickle, percentage, dim, domain, otherDomain):
        self.DATA_FP = data
        self.X_PICKLE = pickle
        self.PERCENTAGE = percentage
        self.DIM = dim
        self.CLUSTERING = True
        self.CLIC_UNSTRUCTURED = False
        self.MIN_DIM = 1024
        self.THIS_DOMAIN = domain
        self.OTHER_DOMAINS = otherDomain
        self.OVERLAPPING_REGIONS = None
        self.IDX_MESH_TO_IDX_NETWORK = None
        self.IDX_MESH_TO_IDX_NETWORK_MATCH = None

    def getDataDir(self):
        return self.DATA_FP
    
    def getPercentOfVertices(self):
        return self.PERCENTAGE

    def getDim(self):
        return self.DIM

    def getMinDim(self):
        return self.MIN_DIM
    
    def getOtherSubdomainFP(self):
        path = self.DATA_FP
        pathDir, _ = path.split("subdomain_")
        if len(self.OTHER_DOMAINS) != 1:
            raise NotImplementedError("Not Implemented more than 2 subdomains in Big Data Assimilation")
        else:
            fp = pathDir + "subdomain_" + str(self.OTHER_DOMAINS[0]) + "/"
            fp = fp + "LSBU_0_" + str(self.OTHER_DOMAINS[0]) + ".vtu"
        return fp
    
    def setOverlappingRegions(self, indices):
        self.OVERLAPPING_REGIONS = indices
    
    def setIdxMeshToIdxNetwork(self, idxDict):
        self.IDX_MESH_TO_IDX_NETWORK = idxDict

    def setMatchIdxMeshToIdxNetwork(self, idxDict):
        self.IDX_MESH_TO_IDX_NETWORK_MATCH = idxDict
