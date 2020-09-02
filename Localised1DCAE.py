"""
Unstructured Mesh CAE: Localisation by Clustering and 1D CAE
Train Tucodec model in 1D from Julian Mack 3D's Implementation: http://www.sciencedirect.com/science/article/pii/S004578252030476X
Train for two block types:
    blocks = ["NeXt"]
    Cstd = [64]
"""

import sys, os, argparse
from datetime import datetime

# Append path to parent directory
sys.path.append("/Users/maxime/IndividualProject/code/Data_Assimilation/src/")

from UnstructuredCAEDA.settings.models.resNeXt import ResStack3
from UnstructuredCAEDA.train import TrainAE
from UnstructuredCAEDA.VarDA import BatchDA
from UnstructuredCAEDA.train import retrain

from UnstructuredCAEDA.UnstructuredMesh.DataLoaderUnstructuredMesh import *
from UnstructuredCAEDA.UnstructuredMesh.CLICUnstructuredMesh import *
from UnstructuredCAEDA.UnstructuredMesh.HelpersUnstructuredMesh import *
from UnstructuredCAEDA.UnstructuredMesh.LocalisationLoader import *
from UnstructuredCAEDA.utils.expt_config import ExptConfigTest

TEST = False
GPU_DEVICE = 0
exp_base = ""

##global variables for DA and training:
class ExptConfig():
    EPOCHS = 20
    SMALL_DEBUG_DOM = False #For training
    calc_DA_MAE = False
    num_epochs_cv = 0
    LR = 0.0002
    print_every = 10
    test_every = 10

def oneSubdomainTrainingAndDA(dataDir, pickledDir, percentage, modelName, dim, thisDomain, otherDomains, title, GPU, TEST, expt, retrain, meanHist, obsVariance):

    loaderSettings = LoaderSetting(dataDir, pickledDir, percentage, dim, thisDomain, otherDomains)
    loader = DataLoaderUnstructuredMesh()
    X = loader.get_X(loaderSettings)

    CLIC_kwargs = { "model_name": "Tucodec1D", "block_type": "NeXt1D", "Cstd": 64,
                    "loader": DataLoaderUnstructuredMesh, "dim": dim, "clusterInputSize": np.shape(X)[1], "name": modelName}
    opt_kwargs = {  "percentage": percentage, 
                    "1D": True, "2D": False, 
                    "thisDomain": thisDomain, 
                    "otherDomains": otherDomains, 
                    "daBackgroundIdx": 1, 
                    "compressionMethod": "AE",
                    "reducedSpace": True, 
                    "backgroundIsMeanHistoricalData": meanHist, 
                    "obsVariance": obsVariance,
                    "modelName": modelName}

    settings = CLICUnstructuredMesh(CLIC_kwargs, opt_kwargs)
    
    settings.setXPickledPath(pickledDir)
    settings.setDataPath(dataDir)
    settings.setExpTitle(title)
    settings.GPU_DEVICE = GPU
    getSettingsInfo(settings)

    expdir = "/experiments/CAE/" + title + "/" + str(settings.getSubdomain())

    settings.setExptDir(expdir)
    
    if retrain:
        newExpDir = "/experiments/CAE/retrain_" + title + "/" + str(settings.getSubdomain()) 
    else:  
        trainer = TrainAE(settings, expdir)
    
    print("------------------------------ Training subdomain {} at {}. ------------------------------".format(settings.getSubdomain(), datetime.now()))
    expdir = trainer.expdir #get full path
    model = trainer.train(num_epochs=expt.EPOCHS, num_workers=0, test_every=expt.test_every,
                        num_epochs_cv=expt.num_epochs_cv,
                        learning_rate = expt.LR,
                        small_debug=expt.SMALL_DEBUG_DOM,
                        calc_DA_MAE=expt.calc_DA_MAE)   #this will take approximately 8 hrs on a K80

    print("------------------------------ DA subdomain {} at {}. ------------------------------".format(settings.getSubdomain(), datetime.now()))
    csvFp = expdir + "/results.csv"
    csvAvgFp = expdir + "/resultsAvg.csv"
    results_df8 = BatchDA(settings, AEModel=model, csv_fp=csvFp, csv_avg_fp=csvAvgFp, save_vtu=True).run()
    print("Results of DA at {}. \n {}".format(datetime.now(), results_df8))

def main():
    cwd = os.getcwd()

    print("\n------------------------- Simulation Started at {} ------------------- \n".format(datetime.now()))

    parser = argparse.ArgumentParser(description="Unstructured Convolutional Autoencoders for Big Data Assimilation")
    parser.add_argument('-d', '--all_data_dir', nargs="*", help='path where the one subdomain data is stored')
    parser.add_argument('-pick', '--pickled_dir', help='path where pickled X are to be stored')
    parser.add_argument('-t', '--title', help='title for experiment being run')
    parser.add_argument('-e', '--exp_dir', help='path where experiment results are to be stored')
    parser.add_argument('-perc', '--percentage', help='percentage of relevant locations chosen for experiment', type=int)
    parser.add_argument('-n', '--model_name', help='specify name of model to use (1D2L, 1D4L, 1D0L)')
    parser.add_argument('-r', '--retrain', help='retrain model if required and saves results to new experiment directory', type=int)
    parser.add_argument('-x', '--dimensions', help='dimension of problem', type=int)
    parser.add_argument('-m', '--mean_hist_data', help='take background state to be mean of historical data')
    parser.add_argument('-o', '--observation_variance', help='defines observational variance for observation covariance matrix R in DA', type=float)
    args = parser.parse_args()

    if TEST:
        expt = ExptConfigTest()
    else:
        expt = ExptConfig()

    for i in range(len(args.all_data_dir)):
        dataDir = args.all_data_dir[i]
        thisDomain = getSubdomainFromFP(dataDir)
        otherDomains = getOtherSubdomainsFromFp(args.all_data_dir, i)
        pickledDir = getThisDomainPickleDir(args.pickled_dir, thisDomain)
        oneSubdomainTrainingAndDA(dataDir, pickledDir, args.percentage, args.model_name, args.dimensions, thisDomain, otherDomains, args.title, GPU_DEVICE, TEST, expt, args.retrain, args.mean_hist_data, args.observation_variance)
        os.chdir(cwd)

    print("------------------------- Ended at {} -------------------- \n".format(datetime.now()))

if __name__ == "__main__":
    main()


