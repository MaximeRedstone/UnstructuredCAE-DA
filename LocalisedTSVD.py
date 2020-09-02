"""
Unstructured TSVD: Localisation by Clustering and TSVD 
"""

import sys, os, argparse

#Append path to parent directory
sys.path.append("/Users/maxime/IndividualProject/code/Data_Assimilation/src/")

from UnstructuredCAEDA.VarDA import *
from UnstructuredCAEDA.settings import Config

from UnstructuredCAEDA.UnstructuredMesh.CLICUnstructuredMesh import CLICUnstructuredMesh
from UnstructuredCAEDA.UnstructuredMesh.DataLoaderUnstructuredMesh import *
from UnstructuredCAEDA.UnstructuredMesh.HelpersUnstructuredMesh import *

def TSVDOneSubdomain(dataDir, pickledDir, percentage, dim, thisDomain, otherDomains, title, GPU, meanHist, obsVariance, modelName = "TSVD"):

    CLIC_kwargs = { "model_name": "Tucodec1D", "block_type": "NeXt1D", "Cstd": 64, 
                    "loader": DataLoaderUnstructuredMesh, "dim": dim, "clusterInputSize": None, "name": modelName}
    opt_kwargs = {  "percentage": percentage, 
                    "1D": True, "2D": False, 
                    "thisDomain": thisDomain, 
                    "otherDomains": otherDomains, 
                    "daBackgroundIdx": 10, 
                    "compressionMethod": "SVD", 
                    "reducedSpace": False, 
                    "backgroundIsMeanHistoricalData": meanHist, 
                    "obsVariance": obsVariance,
                    "modelName": modelName}
    settings = CLICUnstructuredMesh(CLIC_kwargs, opt_kwargs)
    
    settings.setXPickledPath(pickledDir)
    settings.setDataPath(dataDir)
    settings.setClustering(True)
    settings.GPU_DEVICE = GPU

    expdir = "/experiments/TSVD/" + title + "/" + str(settings.getSubdomain())
    settings.setExptDir(expdir)

    csvFp = expdir + "/results.csv"
    csvAvgFp = expdir + "/resultsAvg.csv"

    print("------------------------------ Subdomain {} TSVD at {} ------------------------------".format(settings.getSubdomain(), datetime.now()))

    DA = BatchDA(settings, csv_fp=csvFp, csv_avg_fp=csvAvgFp, save_vtu=True).run()

def main():

    GPU_DEVICE = 0
    cwd = os.getcwd()

    parser = argparse.ArgumentParser(description="Unstructured Convolutional Autoencoders for Big Data Assimilation")
    parser.add_argument('-d', '--all_data_dir', nargs="*", help='path where the one subdomain data is stored')
    parser.add_argument('-pick', '--pickled_dir', help='path where pickled X are to be stored')
    parser.add_argument('-t', '--title', help='title for experiment being run')
    parser.add_argument('-e', '--exp_dir', help='path where experiment results are to be stored')
    parser.add_argument('-perc', '--percentage', help='percentage of relevant locations chosen for experiment', type=int)
    parser.add_argument('-r', '--retrain', help='retrain model if required and saves results to new experiment directory', type=int)
    parser.add_argument('-x', '--dimensions', help='dimension of problem', type=int)
    parser.add_argument('-m', '--mean_hist_data', help='take background state to be mean of historical data'),
    parser.add_argument('-o', '--observation_variance', help='choose background variance for background error covariance matrix', type=float)
    args = parser.parse_args()

    for i in range(len(args.all_data_dir)):
        dataDir = args.all_data_dir[i]
        print("data fp ", dataDir)
        thisDomain = getSubdomainFromFP(dataDir)
        otherDomains = getOtherSubdomainsFromFp(args.all_data_dir, i)
        pickledDir = getThisDomainPickleDir(args.pickled_dir, thisDomain)
        print("domain, other, pickle, dim", thisDomain, otherDomains, pickledDir, args.dimensions)
        TSVDOneSubdomain(dataDir, pickledDir, args.percentage, args.dimensions, thisDomain, otherDomains, args.title, GPU_DEVICE, args.mean_hist_data, args.observation_variance)
        os.chdir(cwd)

    print("------------------------------ End Simulations at {} ------------------------------".format(datetime.now()))


if __name__ == "__main__":   
    main()
